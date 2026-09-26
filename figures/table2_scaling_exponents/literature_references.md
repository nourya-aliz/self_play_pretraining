# Published scaling-law exponents by modality

Verified = read from the paper's own text/tables. Fit forms and units noted per entry;
most literature exponents are vs N (params) or in nats/token, not directly bits/byte against compute.

## Text
- kaplan2020scaling (arXiv:2001.08361, VERIFIED): L(C_min) = (C_c/C)^0.050 (compute envelope,
  no offset); L(N) ~ N^-0.076, L(D) ~ D^-0.095. nats/token, BPE, WebText2.
- hoffmann2022training (Chinchilla, arXiv:2203.15556, VERIFIED): L = 1.69 + 406.4/N^0.34 + 410.7/D^0.28.
- henighan2020scaling language row: alpha_C = 0.048 (nats/token, no offset). VERIFIED.
- hestness2017deep (arXiv:1712.00409, VERIFIED): char-level RHN loss ~ D^-0.094 (data scaling, LSTM-era).
- Byte-level bpb-vs-C fits: NONE published. BLT (pagnoni2024byte, arXiv:2412.09871) plots bpb-vs-FLOPs
  curves but fits no exponent (verified). enwik8/9: none.

## Images (AR pixel/VQ)
- henighan2020scaling Table 1 (VERIFIED), L = L_inf + (C0/C)^alpha_C, nats/token:
  8x8: 0.19 (L_inf 3.13) | 16x16: 0.16 (2.64) | 32x32: 0.10 (2.21) | VQ16: 0.11 (4.09) | VQ32: 0.12 (3.17)
  Web images, not CIFAR-10. No newer AR pixel-level compute fits found.
- aghajanyan2023scaling image row (VQ tokens): alpha=0.13, beta=0.13, E=2.84.

## Audio
- cuervo2024scaling (arXiv:2404.00685, VERIFIED, HuBERT units 25Hz): Chinchilla form
  E=1.73, A=13.9, B=39.8, alpha=0.25, beta=0.24. Compute envelope L = 4.83*C^-0.02 (NO offset,
  near saturation, hence the tiny exponent).
- aghajanyan2023scaling speech row (HuBERT 50Hz): alpha=0.31, beta=0.24, E=3.02. VERIFIED.
- manakul2026soda (arXiv:2602.16687, VERIFIED): Mimi codec audio; allocation N* ~ C^0.367,
  D* ~ C^0.579; loss-vs-C exponent not stated.
- Music audio (MusicNet), environmental sound: NONE. droppo2021acoustic (arXiv:2106.09488) is
  ASR data-scaling only (secondary source only).

## Symbolic music / MIDI
- qu2024mupt (arXiv:2404.06393, VERIFIED w/ caveat): "SMS Law" for ABC notation; numeric
  alpha/beta/E NOT in main text. MIDI-specific: NONE.

## Math / formal
- henighan2020scaling math row: alpha_C = 0.17, L_inf = 0.14 (procedural math). VERIFIED.
- Metamath/Lean loss exponents: NONE (GPT-f etc. report pass rates, no fits).

## DNA / genomics
- nguyen2024evo (Science 386, VERIFIED w/ caveat): compute-optimal DNA scaling analysis,
  single-nucleotide; numeric exponents NOT in main text (figures only; check supplement).
- shah2026dnahnet (arXiv:2602.10603, VERIFIED): PPL = A*C^-alpha on compute-optimal configs,
  nucleotide-level, GTDB genomes: alpha = 0.06 (dnaHNet), 0.04 (StripedHyena2/Evo-2 arch),
  0.01 (Transformer++). bits/base = log2 PPL.

## Code
- aghajanyan2023scaling code row (InCoder data, BPE): E=0.16, alpha=0.37, beta=0.32. VERIFIED.
- arXiv:2510.08702 "Scaling Laws for Code: A More Data-Hungry Regime" (VERIFIED; fetch authors
  before citing): E=0.2193, alpha=0.4853, beta=0.2983, GitHub code.
- Pointer (not fetched): arXiv:2512.13472 "Scaling Laws for Code: Every Programming Language Matters".

## Confirmed gaps (safe "--")
byte-level text bpb-vs-C; CIFAR-10-specific AR exponent (Henighan 32x32 = nearest proxy);
MIDI; MusicNet / environmental sound; Metamath/formal; Evo's numeric DNA exponent (use dnaHNet).

Full source links: Henighan arxiv.org/html/2010.14701v2; Kaplan/Hoffmann/Hestness on ar5iv;
Cuervo aclanthology.org/2024.emnlp-main.21.pdf; Aghajanyan arxiv.org/pdf/2301.03728;
SODA 2602.16687; BLT 2412.09871; MuPT 2404.06393; Evo PMC12057570; dnaHNet 2602.10603.
