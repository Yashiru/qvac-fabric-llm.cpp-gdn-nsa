#include "nsa.cuh"

#define NSA_SEL_WARPS 8  // warps par bloc nsa_sel : 8 x 32 = 256 threads se partagent les cles d'une requete

static __device__ __forceinline__ float nsa_to_f32(const float x) { return x; }                // cle ou valeur F32 : lecture directe, zero conversion
static __device__ __forceinline__ float nsa_to_f32(const half x)  { return __half2float(x); }  // cache KV F16 : conversion en registre, calcul en F32

// NSA, branche compressee + scores de blocs (ggml_nsa_cmp, voir ggml.h).
// Mapping   : 1 bloc CUDA = 1 couple (tete kv hk, token i) ; 1 warp = 1 tete q du groupe GQA.
// Strategie : chaque warp fait un softmax sur les blocs complets visibles (lanes = dims de D,
//             score d'un bloc par reduction de warp) et range ses probas normalisees en SMEM ;
//             la somme sur les G tetes du groupe se fait dans un ordre fixe : resultat deterministe.
// Contraintes : D multiple de 32 ; G = n_head / n_head_kv <= 32 ; SMEM dynamique G * n_blk floats.
template <int D>
__global__ void __launch_bounds__(1024) nsa_cmp_kernel(
        const float * __restrict__ q, const float * __restrict__ kc, const float * __restrict__ vc,
        const int32_t * __restrict__ pos, float * __restrict__ dst,
        const int64_t sq1, const int64_t sq2, const int64_t sk1, const int64_t sk2, const int64_t sv1, const int64_t sv2,
        const int n_head, const int n_head_kv, const int n_tokens, const int n_blk, const int bs, const float scale) {
    constexpr int R = D / WARP_SIZE;  // dims de D tenues par chaque lane : 2 pour D = 64, 4 pour D = 128

    const int hk   = blockIdx.x;               // tete kv du bloc CUDA, plage [0, n_head_kv)
    const int i    = blockIdx.y;               // token de l'ubatch traite par le bloc CUDA, plage [0, n_tokens)
    const int warp = threadIdx.x / WARP_SIZE;  // rang du warp dans le bloc = rang g de la tete q dans le groupe
    const int lane = threadIdx.x % WARP_SIZE;  // lane : possede les dims lane, lane + 32, ... de D
    const int G    = blockDim.x / WARP_SIZE;   // tetes q par tete kv : un warp lance par tete du groupe
    const int h    = hk * G + warp;            // tete q globale : tetes [hk*G, hk*G + G) lisent la tete kv hk

    extern __shared__ float smem[];   // SMEM dynamique : G lignes de n_blk probas, une ligne par warp
    float * p = smem + warp * n_blk;  // ligne de probas du warp : jamais ecrite par un autre warp

    const int t     = pos[i];                    // position absolue t de la requete dans la sequence
    const int ic    = t / bs;                    // bloc courant contenant t, incomplet sauf si t + 1 multiple de bs
    const int n_vis = min((t + 1) / bs, n_blk);  // blocs complets visibles : seuls candidats du softmax compresse

    float qr[R];  // q de la tete h en registres, deja mis a l'echelle
#pragma unroll
    for (int r = 0; r < R; ++r) {  // boucle deroulee sur les R dims de q possedees par la lane
        qr[r] = q[i * sq2 + h * sq1 + lane + r * WARP_SIZE] * scale;  // lecture coalescee : 32 lanes = 32 floats contigus
    }

    float acc[R] = { 0.0f };  // accumulateur registre de o_cmp : reste nul si aucun bloc complet (t < bs - 1)

    if (n_vis > 0) {  // au moins un bloc complet : sinon o_cmp reste nul comme la reference
        float m = -INFINITY;               // max courant des scores : stabilise exp() contre l'overflow F32
        for (int c = 0; c < n_vis; ++c) {  // un score par bloc complet visible, calcule par tout le warp
            const float * kr = kc + c * sk2 + hk * sk1;  // moyenne des cles du bloc c pour la tete kv hk, D floats contigus
            float part = 0.0f;                           // produit scalaire partiel sur les R dims de la lane
#pragma unroll
            for (int r = 0; r < R; ++r) {  // deroulee : R FMA independants par lane, aucune boucle a l'execution
                part += qr[r] * kr[lane + r * WARP_SIZE];  // FMA registre, lecture coalescee de la moyenne de bloc
            }
            const float s = warp_reduce_sum(part);  // score complet du bloc c, identique dans les 32 lanes (xor shuffle)
            if (lane == 0) {                        // lane 0 seule publie le score brut du bloc c en SMEM
                p[c] = s;  // une seule lane ecrit le score brut : evite 32 ecritures SMEM
            }
            m = fmaxf(m, s);  // max uniforme dans le warp : aucune divergence de branche
        }
        __syncwarp();  // barriere warp : scores bruts en SMEM avant relecture par toutes les lanes

        float sum = 0.0f;                                // somme partielle des exponentielles, lanes en pas de 32
        for (int c = lane; c < n_vis; c += WARP_SIZE) {  // lanes en pas de 32 : chaque proba de bloc a une lane proprietaire
            const float e = expf(p[c] - m);  // exponentielle stable : argument <= 0 car m est le max
            p[c] = e;                        // proba non normalisee, ecrite par la lane proprietaire de c
            sum += e;                        // accumulation locale de la normalisation du softmax
        }
        sum = warp_reduce_sum(sum);                      // normalisation complete du softmax, identique dans les 32 lanes
        const float inv = 1.0f / sum;                    // une division par requete : les blocs utilisent une multiplication
        for (int c = lane; c < n_vis; c += WARP_SIZE) {  // lanes en pas de 32 : chaque lane normalise ses propres probas
            p[c] *= inv;  // proba normalisee du bloc c : relue par la somme de groupe
        }
        __syncwarp();  // barriere warp : probas normalisees visibles avant le cumul de o_cmp

        for (int c = 0; c < n_vis; ++c) {  // cumul de o_cmp sur les blocs visibles, c croissant comme le kernel CPU
            const float pc = p[c];                       // meme adresse SMEM pour 32 lanes : broadcast sans bank conflict
            const float * vr = vc + c * sv2 + hk * sv1;  // moyenne des valeurs du bloc c pour la tete kv hk
#pragma unroll
            for (int r = 0; r < R; ++r) {  // deroulee : R FMA par lane sur la valeur moyenne du bloc c
                acc[r] += pc * vr[lane + r * WARP_SIZE];  // o_cmp += p_c * vbar_c, dims de la lane, lecture coalescee
            }
        }
    }

    float * o = dst + ((int64_t) i * n_head + h) * D;  // plan o_cmp de la sortie 1D : ligne (token i, tete h), D floats
#pragma unroll
    for (int r = 0; r < R; ++r) {  // deroulee : R ecritures par lane, une ligne de D floats par warp
        o[lane + r * WARP_SIZE] = acc[r];  // ecriture coalescee de o_cmp, une ligne de D floats par warp
    }

    __syncthreads();  // barriere bloc : lignes de probas des G warps completes avant la somme de groupe

    float * scores = dst + (int64_t) n_tokens * n_head * D;  // plan scores : demarre juste apres le plan o_cmp (n_tokens*n_head*D)
    for (int c = threadIdx.x; c < n_blk; c += blockDim.x) {  // threads du bloc en pas de blockDim : un score de bloc par thread
        float v = 0.0f;   // score du bloc c pour la tete kv hk : somme des probas du groupe
        if (c < n_vis) {  // seuls les blocs visibles ont des probas valides en SMEM
            for (int w = 0; w < G; ++w) {  // parcours des G lignes de probas, une par tete q du groupe
                v += smem[w * n_blk + c];  // somme en ordre fixe w = 0..G-1 : meme ordre que le kernel CPU
            }
        }
        if (c == 0 || c == ic || c == ic - 1) {  // blocs forces : bloc 0, bloc courant ic, bloc precedent ic - 1
            v = INFINITY;  // bloc 0, courant ou precedent : toujours pris par ggml_top_k
        } else if (c >= n_vis) {  // bloc invisible non force : exclu du choix de ggml_top_k
            v = -INFINITY;  // bloc futur non force : jamais prefere a un bloc visible
        }
        scores[((int64_t) i * n_head_kv + hk) * n_blk + c] = v;  // ligne de scores (token i, tete kv hk), lue par ggml_top_k
    }
}

// NSA, branches selection (plan 0) et fenetre glissante (plan 1) (ggml_nsa_sel, voir ggml.h).
// Mapping   : 1 bloc CUDA = 1 triplet (tete q h, token i, plan) ; NSA_SEL_WARPS warps par bloc.
// Strategie : chaque warp parcourt une cle sur NSA_SEL_WARPS avec un softmax en ligne (lanes = dims
//             de D, score par reduction de warp), puis le warp 0 fusionne les etats (m, s, acc).
// Contraintes : D multiple de 32 ; la cle j est a la position j ; les cles > t ne sont jamais lues,
//               donc une cellule perimee (NaN eventuel) du cache ne peut pas contaminer la sortie.
template <int D, typename KT, typename VT>
__global__ void __launch_bounds__(NSA_SEL_WARPS * WARP_SIZE) nsa_sel_kernel(
        const float * __restrict__ q, const KT * __restrict__ k, const VT * __restrict__ v,
        const int32_t * __restrict__ idx, const int32_t * __restrict__ pos, float * __restrict__ dst,
        const int64_t sq1, const int64_t sq2, const int64_t sk1, const int64_t sk2, const int64_t sv1, const int64_t sv2,
        const int64_t si1, const int64_t si2, const int n_head, const int n_head_kv, const int n_tokens,
        const int n_kv, const int n_sel, const int bs, const int window, const float scale) {
    constexpr int R = D / WARP_SIZE;  // dims de D tenues par chaque lane : 2 pour D = 64, 4 pour D = 128

    const int h     = blockIdx.x;                // tete q du bloc CUDA, plage [0, n_head)
    const int i     = blockIdx.y;                // token de l'ubatch traite par le bloc CUDA
    const int plane = blockIdx.z;                // 0 = blocs choisis par top_k, 1 = fenetre [t - window + 1, t]
    const int warp  = threadIdx.x / WARP_SIZE;   // rang du warp : traite les cles j0 + warp + k * NSA_SEL_WARPS
    const int lane  = threadIdx.x % WARP_SIZE;   // lane : possede les dims lane, lane + 32, ... de D
    const int hk    = h / (n_head / n_head_kv);  // tete kv lue par la tete q h : convention GQA de repeat_interleave

    __shared__ float sm[NSA_SEL_WARPS];       // max final m de chaque warp, pour la fusion par le warp 0
    __shared__ float ss[NSA_SEL_WARPS];       // somme finale s de chaque warp, exprimee relativement a son m
    __shared__ float sacc[NSA_SEL_WARPS][D];  // accumulateur final de chaque warp : 8 x D floats, lanes contigues

    const int t  = pos[i];            // position absolue t de la requete
    const int je = min(t + 1, n_kv);  // borne de fin exclusive des cles lisibles : causalite j <= t

    float qr[R];  // q de la tete h en registres, deja mis a l'echelle
#pragma unroll
    for (int r = 0; r < R; ++r) {  // boucle deroulee sur les R dims de q possedees par la lane
        qr[r] = q[i * sq2 + h * sq1 + lane + r * WARP_SIZE] * scale;  // lecture coalescee : 32 lanes = 32 floats contigus
    }

    float m = -INFINITY;      // max courant des scores du warp : premiere cle remet s et acc a l'echelle
    float s = 0.0f;           // somme courante des exp(score - m) du warp
    float acc[R] = { 0.0f };  // somme courante des exp(score - m) * v, dims de la lane

    auto add_key = [&](const int j) {  // lambda device : ajoute la cle j au softmax en ligne du warp
        const KT * kr = k + (int64_t) j * sk2 + hk * sk1;  // cle j de la tete kv hk dans la vue du cache, D elements
        float part = 0.0f;                                 // produit scalaire partiel sur les R dims de la lane
#pragma unroll
        for (int r = 0; r < R; ++r) {  // deroulee : R FMA independants par lane sur la ligne de cle j
            part += qr[r] * nsa_to_f32(kr[lane + r * WARP_SIZE]);  // FMA registre, lecture coalescee de la ligne de cle
        }
        const float score = warp_reduce_sum(part);  // score complet de la cle j, identique dans les 32 lanes
        if (score > m) {                            // nouveau max : etat courant (s, acc) a ramener au nouveau m
            const float c = expf(m - score);  // facteur de reechelonnage vers le nouveau max, 0 au premier passage
            s *= c;                           // somme exprimee relativement au nouveau max
#pragma unroll
            for (int r = 0; r < R; ++r) {  // deroulee : reechelonnage des R dims de l'accumulateur de la lane
                acc[r] *= c;  // accumulateur exprime relativement au nouveau max
            }
            m = score;  // nouveau max, uniforme dans le warp : pas de divergence
        }
        const float w = expf(score - m);                   // poids non normalise de la cle j, argument <= 0
        s += w;                                            // normalisation courante du softmax en ligne
        const VT * vr = v + (int64_t) j * sv2 + hk * sv1;  // valeur j de la tete kv hk, cache V non transpose
#pragma unroll
        for (int r = 0; r < R; ++r) {  // deroulee : R FMA par lane sur la ligne de valeur j
            acc[r] += w * nsa_to_f32(vr[lane + r * WARP_SIZE]);  // FMA registre : contribution ponderee de la valeur j
        }
    };

    if (plane == 0) {  // plan 0 : blocs choisis ; la fenetre glissante est le plan 1
        const int32_t * bi = idx + (int64_t) i * si2 + hk * si1;  // n_sel indices de blocs choisis, partages par le groupe GQA
        for (int is = 0; is < n_sel; ++is) {                      // parcours des n_sel blocs choisis pour la tete kv hk
            const int b = bi[is];  // indice de bloc choisi, negatif = emplacement vide
            if (b < 0) {           // indice negatif : emplacement de top_k laisse vide
                continue;  // emplacement vide : aucune cle lue, branche uniforme du bloc
            }
            const int j0 = b * bs;                                 // premiere cle du bloc b
            const int j1 = min(j0 + bs, je);                       // fin exclusive coupee a t : bloc futur => j1 <= j0, zero iteration
            for (int j = j0 + warp; j < j1; j += NSA_SEL_WARPS) {  // cles du bloc b entrelacees : warp w lit j0 + w, j0 + w + 8, ...
                add_key(j);  // cle j du bloc b, cles entrelacees entre les 8 warps
            }
        }
    } else {
        const int j0 = max(0, t - window + 1);                 // premiere cle de la fenetre glissante, clampee a 0
        for (int j = j0 + warp; j < je; j += NSA_SEL_WARPS) {  // cles de la fenetre entrelacees : warp w lit j0 + w, j0 + w + 8, ...
            add_key(j);  // cle j de la fenetre, cles entrelacees entre les 8 warps
        }
    }

    if (lane == 0) {  // lane 0 seule publie m et s, valeurs uniformes dans le warp
        sm[warp] = m;  // max du warp publie pour la fusion, -inf si le warp n'a lu aucune cle
        ss[warp] = s;  // somme du warp publiee, relative a sm[warp]
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {  // deroulee : publication des R dims de l'accumulateur de la lane
        sacc[warp][lane + r * WARP_SIZE] = acc[r];  // accumulateur du warp publie : lanes contigues, sans bank conflict
    }
    __syncthreads();  // barriere bloc : etats des 8 warps en SMEM avant la fusion par le warp 0

    if (warp == 0) {  // warp 0 seul fusionne les 8 etats et ecrit la sortie de la branche
        float M = -INFINITY;                       // max global des cles lues par le bloc
        for (int w = 0; w < NSA_SEL_WARPS; ++w) {  // max global M sur les etats publies par les 8 warps
            M = fmaxf(M, sm[w]);  // meme valeur dans les 32 lanes : lecture broadcast
        }
        float S = 0.0f;                            // normalisation globale du softmax de la branche
        float o[R] = { 0.0f };                     // sortie non normalisee de la branche, dims de la lane
        for (int w = 0; w < NSA_SEL_WARPS; ++w) {  // fusion des 8 etats de warp, chacun ramene au max global M
            const float f = sm[w] == -INFINITY ? 0.0f : expf(sm[w] - M);  // poids du warp w ; garde contre -inf - -inf = NaN
            S += ss[w] * f;                                               // somme du warp w ramenee au max global
#pragma unroll
            for (int r = 0; r < R; ++r) {  // deroulee : R dims de l'accumulateur publie par le warp w
                o[r] += sacc[w][lane + r * WARP_SIZE] * f;  // accumulateur du warp w ramene au max global
            }
        }
        const float inv = S > 0.0f ? 1.0f / S : 0.0f;                             // 0 si aucune cle lue (tous les indices a -1)
        float * out = dst + (((int64_t) plane * n_tokens + i) * n_head + h) * D;  // sortie [D, n_head, n_tokens, 2] contigue
#pragma unroll
        for (int r = 0; r < R; ++r) {  // deroulee : R ecritures par lane de la sortie normalisee
            out[lane + r * WARP_SIZE] = o[r] * inv;  // ecriture coalescee de la ligne (plan, token i, tete h)
        }
    }
}

void ggml_cuda_op_nsa_cmp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];  // requetes F32 [D, n_head, n_tokens]
    const ggml_tensor * kc  = dst->src[1];  // moyennes de blocs de cles F32 [D, n_head_kv, n_blk]
    const ggml_tensor * vc  = dst->src[2];  // moyennes de blocs de valeurs F32 [D, n_head_kv, n_blk]
    const ggml_tensor * pos = dst->src[3];  // positions I32 [n_tokens]

    const int   bs    = ggml_get_op_params_i32(dst, 0);  // taille de bloc en cles (64 pour gdn-nsa)
    const float scale = ggml_get_op_params_f32(dst, 1);  // 1/sqrt(D), applique une fois a q dans le kernel

    const int D         = q->ne[0];            // taille de tete, 64 ou 128 (filtre par ggml_cuda_nsa_supported)
    const int n_head    = q->ne[1];            // tetes q
    const int n_tokens  = q->ne[2];            // tokens de l'ubatch : 1 en decodage
    const int n_head_kv = kc->ne[1];           // tetes kv
    const int n_blk     = kc->ne[2];           // blocs du cache, ceil(n_kv / bs)
    const int G         = n_head / n_head_kv;  // tetes q par tete kv = warps par bloc CUDA

    const dim3   grid(n_head_kv, n_tokens);                  // 1 bloc CUDA par couple (tete kv, token)
    const dim3   block(G * WARP_SIZE);                       // 1 warp par tete q du groupe : 512 threads pour G = 16
    const size_t smem = (size_t) G * n_blk * sizeof(float);  // G lignes de probas, 7 Ko pour G = 16 et 28 blocs
    cudaStream_t stream = ctx.stream();                      // flux du backend : ordre garanti avec les ops voisines

    const float * q_d  = (const float *) q->data;       // pointeur device des requetes
    const float * kc_d = (const float *) kc->data;      // pointeur device des moyennes de cles
    const float * vc_d = (const float *) vc->data;      // pointeur device des moyennes de valeurs
    const int32_t * p_d = (const int32_t *) pos->data;  // pointeur device des positions
    float * dst_d = (float *) dst->data;                // sortie 1D : plan o_cmp puis plan scores

    const int64_t sq1 = q->nb[1] / sizeof(float);   // pas entre tetes de q, en floats (vue possible)
    const int64_t sq2 = q->nb[2] / sizeof(float);   // pas entre tokens de q, en floats
    const int64_t sk1 = kc->nb[1] / sizeof(float);  // pas entre tetes kv des moyennes de cles, en floats
    const int64_t sk2 = kc->nb[2] / sizeof(float);  // pas entre blocs des moyennes de cles, en floats
    const int64_t sv1 = vc->nb[1] / sizeof(float);  // pas entre tetes kv des moyennes de valeurs, en floats
    const int64_t sv2 = vc->nb[2] / sizeof(float);  // pas entre blocs des moyennes de valeurs, en floats

    switch (D) {  // D connu a l'execution : une instance de template par taille de tete
        case 64:  // tete de 64 : configuration de gdn-nsa
            nsa_cmp_kernel<64><<<grid, block, smem, stream>>>(  // lancement D = 64 : G warps par couple (tete kv, token)
                q_d, kc_d, vc_d, p_d, dst_d, sq1, sq2, sk1, sk2, sv1, sv2,
                n_head, n_head_kv, n_tokens, n_blk, bs, scale);  // instance D = 64 : 2 dims par lane
            break;  // fin du cas D = 64 : un seul lancement de kernel
        case 128:  // tete de 128 : configuration couverte par test-backend-ops
            nsa_cmp_kernel<128><<<grid, block, smem, stream>>>(  // lancement D = 128 : G warps par couple (tete kv, token)
                q_d, kc_d, vc_d, p_d, dst_d, sq1, sq2, sk1, sk2, sv1, sv2,
                n_head, n_head_kv, n_tokens, n_blk, bs, scale);  // instance D = 128 : 4 dims par lane
            break;  // fin du cas D = 128 : un seul lancement de kernel
        default:  // toute autre taille de tete : refusee en amont par supports_op
            GGML_ABORT("nsa_cmp: unsupported head size %d", D);  // garde : ggml_cuda_nsa_supported n'accepte que 64 et 128
    }
}

template <int D, typename KT, typename VT>
static void nsa_sel_launch(const ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * q   = dst->src[0];  // requetes F32 [D, n_head, n_tokens]
    const ggml_tensor * k   = dst->src[1];  // vue du cache K [D, n_head_kv, n_kv], type KT
    const ggml_tensor * v   = dst->src[2];  // vue du cache V [D, n_head_kv, n_kv], type VT
    const ggml_tensor * idx = dst->src[3];  // indices de blocs I32 [n_sel, n_head_kv, n_tokens]
    const ggml_tensor * pos = dst->src[4];  // positions I32 [n_tokens]

    const int   bs     = ggml_get_op_params_i32(dst, 0);  // taille de bloc en cles
    const int   window = ggml_get_op_params_i32(dst, 1);  // taille de la fenetre glissante en cles (512)
    const float scale  = ggml_get_op_params_f32(dst, 2);  // 1/sqrt(D), applique une fois a q dans le kernel

    const int n_head    = q->ne[1];    // tetes q = blocs CUDA en x
    const int n_tokens  = q->ne[2];    // tokens de l'ubatch = blocs CUDA en y
    const int n_head_kv = k->ne[1];    // tetes kv du cache
    const int n_kv      = k->ne[2];    // cellules du cache visibles par le graphe
    const int n_sel     = idx->ne[0];  // blocs choisis par tete kv, min(16, n_blk)

    const dim3 grid(n_head, n_tokens, 2);                   // z = 2 plans : selection et fenetre en blocs CUDA distincts
    const dim3 block(NSA_SEL_WARPS * WARP_SIZE);            // 256 threads : 8 warps se partagent les cles
    nsa_sel_kernel<D, KT, VT><<<grid, block, 0, stream>>>(  // grille (n_head, n_tokens, 2 plans), 256 threads, 0 SMEM dyn
        (const float *) q->data, (const KT *) k->data, (const VT *) v->data,
        (const int32_t *) idx->data, (const int32_t *) pos->data, (float *) dst->data,
        q->nb[1] / sizeof(float), q->nb[2] / sizeof(float), k->nb[1] / sizeof(KT), k->nb[2] / sizeof(KT),
        v->nb[1] / sizeof(VT), v->nb[2] / sizeof(VT), idx->nb[1] / sizeof(int32_t), idx->nb[2] / sizeof(int32_t),
        n_head, n_head_kv, n_tokens, n_kv, n_sel, bs, window, scale);  // strides en elements : nb[] en octets divise par la taille
}

template <int D>
static void nsa_sel_dispatch_kv(const ggml_tensor * dst, cudaStream_t stream) {
    const bool k16 = dst->src[1]->type == GGML_TYPE_F16;  // cache K en F16 (defaut llama.cpp) ou F32
    const bool v16 = dst->src[2]->type == GGML_TYPE_F16;  // cache V en F16 (defaut llama.cpp) ou F32
    if (k16 && v16) {                                     // K et V en F16 : cache par defaut de llama.cpp
        nsa_sel_launch<D, half, half>(dst, stream);  // cas courant : cache KV F16
    } else if (k16) {  // K seul en F16, V en F32
        nsa_sel_launch<D, half, float>(dst, stream);  // K F16, V F32 (-ctk f16 -ctv f32)
    } else if (v16) {  // V seul en F16, K en F32
        nsa_sel_launch<D, float, half>(dst, stream);  // K F32, V F16 (-ctk f32 -ctv f16)
    } else {
        nsa_sel_launch<D, float, float>(dst, stream);  // cache KV F32 : validation stricte contre la reference
    }
}

void ggml_cuda_op_nsa_sel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int D = dst->src[0]->ne[0];  // taille de tete : choisit l'instance du template
    switch (D) {                       // D connu a l'execution : une instance de template par taille de tete
        case 64:  // tete de 64 : configuration de gdn-nsa
            nsa_sel_dispatch_kv<64>(dst, ctx.stream());  // instance D = 64 : 2 dims par lane
            break;                                       // fin du cas D = 64 : un seul lancement de kernel
        case 128:  // tete de 128 : configuration couverte par test-backend-ops
            nsa_sel_dispatch_kv<128>(dst, ctx.stream());  // instance D = 128 : 4 dims par lane
            break;                                        // fin du cas D = 128 : un seul lancement de kernel
        default:  // toute autre taille de tete : refusee en amont par supports_op
            GGML_ABORT("nsa_sel: unsupported head size %d", D);  // garde : ggml_cuda_nsa_supported n'accepte que 64 et 128
    }
}

bool ggml_cuda_nsa_supported(const ggml_tensor * op) {  // false : le scheduler place l'op sur un autre backend
    const ggml_tensor * q = op->src[0];  // requetes : type et taille de tete imposes par les instances
    const ggml_tensor * k = op->src[1];  // k_cmp (NSA_CMP) ou vue du cache K (NSA_SEL)
    const ggml_tensor * v = op->src[2];  // v_cmp (NSA_CMP) ou vue du cache V (NSA_SEL)

    if (q->type != GGML_TYPE_F32 || (q->ne[0] != 64 && q->ne[0] != 128)) {  // type ou taille de tete hors des instances compilees
        return false;  // instances compilees : q F32 et D = 64 ou 128 seulement
    }
    const int64_t G = q->ne[1] / k->ne[1];     // tetes q par tete kv
    if (q->ne[1] % k->ne[1] != 0 || G > 32) {  // groupe GQA non entier ou plus de 32 tetes q par tete kv
        return false;  // nsa_cmp lance 1 warp par tete du groupe : 32 warps max par bloc
    }
    if (op->op == GGML_OP_NSA_CMP) {  // NSA_CMP : contraintes de types et de SMEM propres a nsa_cmp_kernel
        const bool f32 = k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32;  // moyennes de blocs produites en F32
        return f32 && G * k->ne[2] * sizeof(float) <= 48 * 1024;                // SMEM statique max sans opt-in : 48 Ko
    }
    const bool k_ok = k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32;  // instances nsa_sel : cache F16 ou F32
    const bool v_ok = v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32;  // cache V : F16 ou F32, type independant de celui de K
    return k_ok && v_ok;                                                     // NSA_SEL n'utilise aucune SMEM dynamique
}
