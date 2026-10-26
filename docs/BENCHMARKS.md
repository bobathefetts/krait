# Krait: measured numbers

Every number here was measured; nothing is estimated. They come from one machine, so treat them as one data point.

## Test machine and settings

- GPU: NVIDIA RTX 5090, 32 GB (32,607 MiB), sm_120. Every run used `--vram-limit-mib 29346` (the engine keeps its
  whole VRAM use under 29,346 MiB). Driver 617.42.
- CPU: AMD Ryzen 9 7950X3D, 16 cores / 32 threads; the engine used 16 threads.
- RAM: 127 GB in the PC; the WSL2 VM saw 116,713 MiB (MemTotal 119,514,760 kB). No swap was used.
- Software: Windows 11 Pro + WSL2 (kernel 6.18.33.1) Ubuntu 24.04, CUDA 12.9 (conda-forge), gcc 13.3.
- Model: Unsloth `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL` (3 shards) + `mmproj-BF16.gguf`.
- Engine flags: those of `krait/config.json` (`--n-cpu-moe 68 --cache all --pin --spec mtp --spec-k 1
  --spec-min-acc 0.6 --chunk 512 --max-context 32768`), plus the VRAM cap; `--batch 4` where stated.
- llama.cpp: b11806 (`0682a576a`), built on the same machine, run with the same expert placement (`--n-cpu-moe 68`).
- Decoding: **greedy** (temperature 0) everywhere below. MTP speculation is used only for greedy requests on their own
  (not in `--batch` slots, not for sampled requests); sampled decoding was not benchmarked.
- decode tok/s = (n - 1) / (t_last - t_first), where t is the arrival time of each generated token.

## Decode, real prompts (CLI)

6 chat prompts x 3 runs, 256 tokens each; the median of the 18 runs.

| | median decode tok/s | per-prompt medians (p0, p4, p7, p13, p16, p21) |
|---|---|---|
| Krait | 23.01585 | 22.5289, 22.8486, 22.5867, 23.3631, 23.2167, 23.6092 |
| llama.cpp b11806 (llama-server, same prompts) | 14.300132 | - |
| ratio | 1.6095x | |

## Decode without speculation (earlier build)

Krait's Phase 3 build (VRAM expert cache + pinned host experts, no MTP speculation; flags `--n-cpu-moe 68 --cache all
--pin --chunk 512`, 16 threads, the same VRAM cap), the same 6 prompts x 3 runs, 256 tokens each, greedy, measured in
its own session together with a fresh llama.cpp b11806 run:

| | median decode tok/s | per-prompt medians (p0, p4, p7, p13, p16, p21) |
|---|---|---|
| Krait, no speculation | 20.679 (min 20.18, max 21.46) | 20.296, 20.645, 20.704, 21.146, 20.603, 20.671 |
| llama.cpp b11806, same session | 14.316 | 14.242, 14.328, 14.35, 14.215, 14.308, 14.344 |
| ratio | 1.444x | |

The 18-run median above is from that build. A single run of the 6 prompts after a later expert-cache fix, still
without speculation, gave 20.296 / 20.890 / 21.093 / 21.797 / 20.898 / 21.151 tok/s.

## Prompt reading (synthetic prompts, median of 3)

| prompt tokens | Krait prompt tok/s | Krait time to first token | llama.cpp prompt tok/s | llama.cpp time to first token |
|---|---|---|---|---|
| 1K | 455.04 | 2.250 s | 177.20 | 5.780 s |
| 4K | 510.25 | 8.027 s | 203.77 | 20.103 s |
| 32K (32,512 for llama.cpp) | 512.99 | 63.378 s | 207.29 | 156.849 s |

(llama.cpp's numbers come from the same session as the CLI decode table; an earlier session measured 181.43 / 204.38 /
207.24 tok/s and 156.889 s.)

## MTP speculation (k = 1, greedy)

Mean 0.872 accepted drafts per verify step, i.e. 1.872 tokens committed per verify step; the draft step took 4.17 ms
(4.8% of the decode time).

## Over HTTP (krait/bench_serve.py)

Streaming `/v1/chat/completions`, temperature 0, thinking off, 256 tokens, prompts of 34-41 tokens.

| prompt | time to first token | decode tok/s |
|---|---|---|
| p0 | 0.831 s | 25.0 |
| p1 | 0.773 s | 23.3 |
| p2 | 0.778 s | 24.6 |
| p3 | 0.767 s | 25.8 |
| p4 | 0.861 s | 25.0 |
| p5 | 0.763 s | 26.1 |
| median | | **25.0** |

Several requests at once (`--batch 4`; batch slots decode without MTP), aggregate: 1 request 256 tokens in 10.7 s =
23.92 tok/s; 2 requests 512 tokens in 24.0 s = 21.29 tok/s; 4 requests 1024 tokens in 43.3 s = 23.66 tok/s.

Engine line protocol, concurrent sessions (no MTP in slots), total tok/s: c = 1 / 2 / 3 / 4 -> 19.84 / 22.81 / 23.53 /
23.78 (steady state 19.77 / 23.56 / 24.66 / 25.09). More requests at once do not raise the total: each extra row needs
its own experts, and the misses of the 31 MoE layers in blocks 0-67 run on the CPU.

## Pictures (CPU encoder)

640x480 -> 300 rows in 1.17 s; 640x400 -> 260 rows in 0.98 s.

## Memory

- Expert cache, from the engine's own startup log: `31 lo + 9 hi pools, slots/layer lo 123 hi 123, 17359.6 MiB` alone;
  111 slots per layer (15,666.0 MiB) with `--batch 4`.
- `--pin` page-locked 72,261.3 MiB in 80 ranges.
- Peak VRAM over all runs: 29,287 MiB (cap 29,346). Engine resident set: 81.3-81.8 GiB (85,790,788 kB in the latest
  Gate 2 run). Lowest MemAvailable seen in the VM during the runs: 106,655 MiB. Swap used: 0.

## Exactness checks

| check | result |
|---|---|
| 32 test prompts, greedy, vs the golden capture of a reference tool built on llama.cpp b11806 | 32/32 token-identical; top-1 agreement on 42,075/42,075 logits rows; logits bit-identical (max difference 0), also with MTP on |
| speculation on vs off (`--spec-check`, every logits row hashed) | identical for MTP k = 1, 2, 3 and n-gram k = 1, 2, 3 |
| served greedy tokens vs the CLI | solo 6/6, `--batch 4` 6/6, 1-4 concurrent sessions identical, HTTP 13/13 |
| pictures: engine vs `llama-mtmd-cli` (2 pictures, solo and batched) | 4/4 identical |
| greedy text vs `llama-server` b11806 on 6 real prompts | differs after 40 to 197 tokens |
