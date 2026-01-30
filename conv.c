/* conv.c
   Compile: gcc -O3 conv.c -o conv -lm
   Usage:
     ./conv input.pgm output.pgm kernelname
     ./conv -bench       (runs internal benchmarks and prints times)
   Kernel names: identity, box3, gauss3, sobel, laplacian
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ----------- PGM read/write (binary P5) ---------- */

static void skip_comments(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
        } else {
            ungetc(c, f);
            return;
        }
    }
}

uint32_t *load_pgm(const char *fname, int *W, int *H) {
    FILE *f = fopen(fname, "rb");
    if (!f) { perror("fopen"); return NULL; }
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1) { fclose(f); return NULL; }
    if (strcmp(magic, "P5") != 0) { fprintf(stderr,"Not P5 PGM\n"); fclose(f); return NULL; }
    skip_comments(f);
    int w,h,maxv;
    if (fscanf(f, "%d", &w) != 1) { fclose(f); return NULL; }
    skip_comments(f);
    if (fscanf(f, "%d", &h) != 1) { fclose(f); return NULL; }
    skip_comments(f);
    if (fscanf(f, "%d", &maxv) != 1) { fclose(f); return NULL; }
    fgetc(f); /* consume single whitespace char after maxval */

    *W = w; *H = h;
    size_t npix = (size_t)w * h;
    uint32_t *img = (uint32_t*)malloc(npix * sizeof(uint32_t));
    if (!img) { fclose(f); return NULL; }

    if (maxv < 256) {
        /* 1 byte per pixel */
        unsigned char *buf = (unsigned char*)malloc(npix);
        if (!buf) { free(img); fclose(f); return NULL; }
        if (fread(buf, 1, npix, f) != npix) { perror("read"); free(buf); free(img); fclose(f); return NULL; }
        for (size_t i=0;i<npix;i++) img[i] = (uint32_t)buf[i];
        free(buf);
    } else if (maxv < 65536) {
        /* 2 bytes per pixel (most significant byte first) */
        for (size_t i=0;i<npix;i++) {
            int hi = fgetc(f);
            int lo = fgetc(f);
            if (hi==EOF || lo==EOF) { perror("read"); free(img); fclose(f); return NULL; }
            img[i] = (uint32_t)((hi<<8) | lo);
        }
    } else {
        fprintf(stderr,"Unsupported maxval\n"); free(img); fclose(f); return NULL;
    }
    fclose(f);
    return img;
}

int write_pgm_scaled(const char *fname, const uint32_t *img, int W, int H) {
    size_t np = (size_t)W*H;
    uint32_t minv = UINT32_MAX, maxv = 0;
    for (size_t i=0;i<np;i++) {
        if (img[i] < minv) minv = img[i];
        if (img[i] > maxv) maxv = img[i];
    }
    /* avoid division by zero */
    double range = (double)(maxv - minv);
    if (range < 1.0) range = 1.0;

    FILE *f = fopen(fname, "wb");
    if (!f) { perror("fopen"); return -1; }
    fprintf(f, "P5\n%d %d\n255\n", W, H);

    unsigned char *buf = (unsigned char*)malloc(np);
    if (!buf) { fclose(f); return -1; }
    for (size_t i=0;i<np;i++) {
        double v = (img[i] - (double)minv) / range;
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        buf[i] = (unsigned char) (round(255.0 * v));
    }
    fwrite(buf, 1, np, f);
    free(buf);
    fclose(f);
    return 0;
}

/* ---------- Convolution ---------- */

void convolve_uint32(const uint32_t *in, uint32_t *out, int W, int H,
                     const float *kernel, int K) {
    int kc = K/2;
    for (int y=0;y<H;y++) {
        for (int x=0;x<W;x++) {
            double sum = 0.0;
            for (int ky=0; ky<K; ky++) {
                int iy = y + ky - kc;
                if (iy < 0 || iy >= H) continue; /* zero padding */
                for (int kx=0; kx<K; kx++) {
                    int ix = x + kx - kc;
                    if (ix < 0 || ix >= W) continue;
                    double pixel = (double) in[iy * W + ix];
                    sum += pixel * kernel[ky*K + kx];
                }
            }
            if (sum < 0) out[y*W + x] = 0;
            else if (sum > (double)UINT32_MAX) out[y*W + x] = UINT32_MAX;
            else out[y*W + x] = (uint32_t)(llround(sum));
        }
    }
}

/* Sobel magnitude (two convolutions then magnitude) */
void convolve_sobel_mag(const uint32_t *in, uint32_t *out, int W, int H,
                        const float *kx, const float *ky, int K) {
    uint32_t *tmp = (uint32_t*)malloc((size_t)W*H*sizeof(uint32_t));
    if (!tmp) return;
    convolve_uint32(in, tmp, W, H, kx, K);
    /* using double for magnitude to avoid overflow when squaring */
    int kc = K/2;
    for (int y=0;y<H;y++) {
        for (int x=0;x<W;x++) {
            double gx = (double) tmp[y*W + x];
            /* compute gy by convolving with ky (call convolve into tmp2 or reuse) */
            /* We'll convolve ky on the fly into tmp2 to avoid double storage; do second conv */
        }
    }
    /* do second conv to another buffer */
    uint32_t *tmp2 = (uint32_t*)malloc((size_t)W*H*sizeof(uint32_t));
    if (!tmp2) { free(tmp); return; }
    convolve_uint32(in, tmp2, W, H, ky, K);
    for (size_t i=0;i<(size_t)W*H;i++) {
        double m = sqrt((double)tmp[i]*(double)tmp[i] + (double)tmp2[i]*(double)tmp2[i]);
        if (m < 0) out[i] = 0;
        else if (m > (double)UINT32_MAX) out[i] = UINT32_MAX;
        else out[i] = (uint32_t) llround(m);
    }
    free(tmp); free(tmp2);
}

/* ---------- Kernels ---------- */

float identity3[9] = {0,0,0,0,1,0,0,0,0};
float box3[9] = {1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9};
float gauss3[9] = {1.0f/16,2.0f/16,1.0f/16,2.0f/16,4.0f/16,2.0f/16,1.0f/16,2.0f/16,1.0f/16};
float laplacian3[9] = {0,1,0,1,-4,1,0,1,0};
float sobelx3[9] = {-1,0,1,-2,0,2,-1,0,1};
float sobely3[9] = {-1,-2,-1,0,0,0,1,2,1};

/* ---------- Timing helper ---------- */
double diff_ms(struct timespec a, struct timespec b) {
    return (a.tv_sec - b.tv_sec) * 1000.0 + (a.tv_nsec - b.tv_nsec) / 1e6;
}

/* ---------- Bench: generate random image and measure ---------- */
void bench() {
    int Ms[3] = {512, 1024, 2048}; /* you can change/extend */
    int Ks[3] = {3,5,7};
    printf("M,K,iterations,ms\n");
    for (int mi=0; mi<3; mi++) {
        int M = Ms[mi];
        for (int ki=0; ki<3; ki++) {
            int K = Ks[ki];
            size_t np = (size_t)M*M;
            uint32_t *img = (uint32_t*)malloc(np*sizeof(uint32_t));
            uint32_t *out = (uint32_t*)malloc(np*sizeof(uint32_t));
            if (!img || !out) { fprintf(stderr,"alloc fail\n"); exit(1); }
            srand(12345);
            for (size_t i=0;i<np;i++) img[i] = (uint32_t)(rand() & 0xFF);

            /* make a simple kernel (box KxK normalized) */
            float *kernel = (float*)malloc(K*K*sizeof(float));
            for (int i=0;i<K*K;i++) kernel[i] = 1.0f/(K*K);

            struct timespec t0, t1;
            int iters = 3;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int it=0; it<iters; it++) {
                convolve_uint32(img, out, M, M, kernel, K);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ms = diff_ms(t1,t0);
            printf("%d,%d,%d,%.3f\n", M, K, iters, ms);
            free(img); free(out); free(kernel);
        }
    }
}

/* ---------- Main ---------- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage:\n  %s input.pgm output.pgm kernelname\n  %s -bench\n", argv[0], argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "-bench") == 0) {
        bench();
        return 0;
    }
    if (argc < 4) {
        fprintf(stderr,"need input output kernelname\n");
        return 1;
    }
    const char *infile = argv[1];
    const char *outfile = argv[2];
    const char *kname = argv[3];

    int W,H;
    uint32_t *in = load_pgm(infile, &W, &H);
    if (!in) { fprintf(stderr,"failed to load\n"); return 1; }
    uint32_t *out = (uint32_t*)malloc((size_t)W*H*sizeof(uint32_t));
    if (!out) { free(in); return 1; }

    if (strcmp(kname,"identity")==0) {
        convolve_uint32(in,out,W,H,identity3,3);
    } else if (strcmp(kname,"box3")==0) {
        convolve_uint32(in,out,W,H,box3,3);
    } else if (strcmp(kname,"gauss3")==0) {
        convolve_uint32(in,out,W,H,gauss3,3);
    } else if (strcmp(kname,"laplacian")==0) {
        convolve_uint32(in,out,W,H,laplacian3,3);
    } else if (strcmp(kname,"sobel")==0) {
        convolve_sobel_mag(in,out,W,H,sobelx3,sobely3,3);
    } else {
        fprintf(stderr,"unknown kernel: %s\n", kname);
        free(in); free(out); return 1;
    }

    write_pgm_scaled(outfile, out, W, H);
    free(in); free(out);
    printf("Wrote %s\n", outfile);
    return 0;
}
