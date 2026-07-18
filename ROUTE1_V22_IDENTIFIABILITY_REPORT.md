# Route-1 v2.2 identifiability report

## Accuracy and transfer by task

| pair | method | seed | task | n | accuracy | positive transfer | negative transfer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama32_1b | B0 | 42 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 42 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 42 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 43 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 43 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 43 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 44 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 44 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| llama32_1b | B0 | 44 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| llama32_1b | B1 | 42 | ai2-arc | 1150 | 0.4965 | 0.3474 | 0.2814 |
| llama32_1b | B1 | 42 | mmlu-redux | 5615 | 0.4321 | 0.2678 | 0.2633 |
| llama32_1b | B1 | 42 | openbookqa | 500 | 0.4440 | 0.3344 | 0.3889 |
| llama32_1b | B1 | 43 | ai2-arc | 1150 | 0.5470 | 0.4448 | 0.3009 |
| llama32_1b | B1 | 43 | mmlu-redux | 5615 | 0.4531 | 0.3662 | 0.3859 |
| llama32_1b | B1 | 43 | openbookqa | 500 | 0.4680 | 0.4040 | 0.4343 |
| llama32_1b | B1 | 44 | ai2-arc | 1150 | 0.5487 | 0.4215 | 0.2619 |
| llama32_1b | B1 | 44 | mmlu-redux | 5615 | 0.4431 | 0.3199 | 0.3284 |
| llama32_1b | B1 | 44 | openbookqa | 500 | 0.4920 | 0.4106 | 0.3838 |
| llama32_1b | B2 | 42 | ai2-arc | 1150 | 0.5339 | 0.3735 | 0.2273 |
| llama32_1b | B2 | 42 | mmlu-redux | 5615 | 0.4470 | 0.2928 | 0.2669 |
| llama32_1b | B2 | 42 | openbookqa | 500 | 0.4920 | 0.3510 | 0.2929 |
| llama32_1b | B2 | 43 | ai2-arc | 1150 | 0.5530 | 0.4288 | 0.2619 |
| llama32_1b | B2 | 43 | mmlu-redux | 5615 | 0.4533 | 0.3207 | 0.3010 |
| llama32_1b | B2 | 43 | openbookqa | 500 | 0.5140 | 0.4106 | 0.3283 |
| llama32_1b | B2 | 44 | ai2-arc | 1150 | 0.5209 | 0.3663 | 0.2489 |
| llama32_1b | B2 | 44 | mmlu-redux | 5615 | 0.4420 | 0.2895 | 0.2750 |
| llama32_1b | B2 | 44 | openbookqa | 500 | 0.5020 | 0.3543 | 0.2727 |
| llama32_1b | B3 | 42 | ai2-arc | 1150 | 0.5339 | 0.3735 | 0.2273 |
| llama32_1b | B3 | 42 | mmlu-redux | 5615 | 0.4472 | 0.2930 | 0.2669 |
| llama32_1b | B3 | 42 | openbookqa | 500 | 0.4920 | 0.3510 | 0.2929 |
| llama32_1b | B3 | 43 | ai2-arc | 1150 | 0.5530 | 0.4288 | 0.2619 |
| llama32_1b | B3 | 43 | mmlu-redux | 5615 | 0.4536 | 0.3213 | 0.3010 |
| llama32_1b | B3 | 43 | openbookqa | 500 | 0.5140 | 0.4106 | 0.3283 |
| llama32_1b | B3 | 44 | ai2-arc | 1150 | 0.5209 | 0.3663 | 0.2489 |
| llama32_1b | B3 | 44 | mmlu-redux | 5615 | 0.4420 | 0.2895 | 0.2750 |
| llama32_1b | B3 | 44 | openbookqa | 500 | 0.5020 | 0.3543 | 0.2727 |
| llama32_1b | B5 | 42 | ai2-arc | 1150 | 0.5209 | 0.4172 | 0.3247 |
| llama32_1b | B5 | 42 | mmlu-redux | 5615 | 0.4584 | 0.3451 | 0.3315 |
| llama32_1b | B5 | 42 | openbookqa | 500 | 0.4700 | 0.4073 | 0.4343 |
| llama32_1b | B5 | 43 | ai2-arc | 1150 | 0.5496 | 0.4142 | 0.2489 |
| llama32_1b | B5 | 43 | mmlu-redux | 5615 | 0.4470 | 0.3059 | 0.2913 |
| llama32_1b | B5 | 43 | openbookqa | 500 | 0.4940 | 0.3179 | 0.2374 |
| llama32_1b | B5 | 44 | ai2-arc | 1150 | 0.5261 | 0.3939 | 0.2771 |
| llama32_1b | B5 | 44 | mmlu-redux | 5615 | 0.4524 | 0.3172 | 0.2969 |
| llama32_1b | B5 | 44 | openbookqa | 500 | 0.4700 | 0.3543 | 0.3535 |
| llama32_1b | B6 | 42 | ai2-arc | 1150 | 0.5461 | 0.4084 | 0.2489 |
| llama32_1b | B6 | 42 | mmlu-redux | 5615 | 0.4449 | 0.2887 | 0.2654 |
| llama32_1b | B6 | 42 | openbookqa | 500 | 0.4900 | 0.3642 | 0.3182 |
| llama32_1b | B6 | 43 | ai2-arc | 1150 | 0.5496 | 0.4172 | 0.2532 |
| llama32_1b | B6 | 43 | mmlu-redux | 5615 | 0.4598 | 0.3257 | 0.2913 |
| llama32_1b | B6 | 43 | openbookqa | 500 | 0.5140 | 0.3841 | 0.2879 |
| llama32_1b | B6 | 44 | ai2-arc | 1150 | 0.5426 | 0.3910 | 0.2316 |
| llama32_1b | B6 | 44 | mmlu-redux | 5615 | 0.4363 | 0.2996 | 0.3101 |
| llama32_1b | B6 | 44 | openbookqa | 500 | 0.5220 | 0.4106 | 0.3081 |
| qwen25_0p5b | B0 | 42 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 42 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 42 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 43 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 43 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 43 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 44 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 44 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| qwen25_0p5b | B0 | 44 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| qwen25_0p5b | B1 | 42 | ai2-arc | 1150 | 0.5435 | 0.4259 | 0.2814 |
| qwen25_0p5b | B1 | 42 | mmlu-redux | 5615 | 0.4561 | 0.3522 | 0.3513 |
| qwen25_0p5b | B1 | 42 | openbookqa | 500 | 0.4900 | 0.3841 | 0.3485 |
| qwen25_0p5b | B1 | 43 | ai2-arc | 1150 | 0.5148 | 0.3198 | 0.1948 |
| qwen25_0p5b | B1 | 43 | mmlu-redux | 5615 | 0.4256 | 0.2305 | 0.2125 |
| qwen25_0p5b | B1 | 43 | openbookqa | 500 | 0.4260 | 0.1921 | 0.2172 |
| qwen25_0p5b | B1 | 44 | ai2-arc | 1150 | 0.4687 | 0.3270 | 0.3203 |
| qwen25_0p5b | B1 | 44 | mmlu-redux | 5615 | 0.3995 | 0.2788 | 0.3767 |
| qwen25_0p5b | B1 | 44 | openbookqa | 500 | 0.3940 | 0.2748 | 0.4242 |
| qwen25_0p5b | B2 | 42 | ai2-arc | 1150 | 0.5357 | 0.3794 | 0.2316 |
| qwen25_0p5b | B2 | 42 | mmlu-redux | 5615 | 0.4419 | 0.2752 | 0.2491 |
| qwen25_0p5b | B2 | 42 | openbookqa | 500 | 0.4940 | 0.3477 | 0.2828 |
| qwen25_0p5b | B2 | 43 | ai2-arc | 1150 | 0.5243 | 0.2907 | 0.1277 |
| qwen25_0p5b | B2 | 43 | mmlu-redux | 5615 | 0.4260 | 0.1987 | 0.1525 |
| qwen25_0p5b | B2 | 43 | openbookqa | 500 | 0.4420 | 0.1523 | 0.1162 |
| qwen25_0p5b | B2 | 44 | ai2-arc | 1150 | 0.5122 | 0.2805 | 0.1429 |
| qwen25_0p5b | B2 | 44 | mmlu-redux | 5615 | 0.4346 | 0.1916 | 0.1149 |
| qwen25_0p5b | B2 | 44 | openbookqa | 500 | 0.4860 | 0.2517 | 0.1566 |
| qwen25_0p5b | B3 | 42 | ai2-arc | 1150 | 0.5426 | 0.3445 | 0.1623 |
| qwen25_0p5b | B3 | 42 | mmlu-redux | 5615 | 0.4267 | 0.2352 | 0.2181 |
| qwen25_0p5b | B3 | 42 | openbookqa | 500 | 0.4820 | 0.2517 | 0.1667 |
| qwen25_0p5b | B3 | 43 | ai2-arc | 1150 | 0.5339 | 0.3663 | 0.2165 |
| qwen25_0p5b | B3 | 43 | mmlu-redux | 5615 | 0.4433 | 0.2939 | 0.2796 |
| qwen25_0p5b | B3 | 43 | openbookqa | 500 | 0.4640 | 0.2649 | 0.2323 |
| qwen25_0p5b | B3 | 44 | ai2-arc | 1150 | 0.5078 | 0.2703 | 0.1385 |
| qwen25_0p5b | B3 | 44 | mmlu-redux | 5615 | 0.4281 | 0.1916 | 0.1332 |
| qwen25_0p5b | B3 | 44 | openbookqa | 500 | 0.4660 | 0.2384 | 0.1869 |
| qwen25_0p5b | B5 | 42 | ai2-arc | 1150 | 0.5078 | 0.3096 | 0.1970 |
| qwen25_0p5b | B5 | 42 | mmlu-redux | 5615 | 0.4256 | 0.2410 | 0.2318 |
| qwen25_0p5b | B5 | 42 | openbookqa | 500 | 0.4840 | 0.2848 | 0.2121 |
| qwen25_0p5b | B5 | 43 | ai2-arc | 1150 | 0.5557 | 0.3823 | 0.1861 |
| qwen25_0p5b | B5 | 43 | mmlu-redux | 5615 | 0.4522 | 0.2738 | 0.2171 |
| qwen25_0p5b | B5 | 43 | openbookqa | 500 | 0.4960 | 0.3046 | 0.2121 |
| qwen25_0p5b | B5 | 44 | ai2-arc | 1150 | 0.4809 | 0.2297 | 0.1450 |
| qwen25_0p5b | B5 | 44 | mmlu-redux | 5615 | 0.4119 | 0.1842 | 0.1657 |
| qwen25_0p5b | B5 | 44 | openbookqa | 500 | 0.4800 | 0.2450 | 0.1616 |
| qwen25_0p5b | B6 | 42 | ai2-arc | 1150 | 0.5357 | 0.3430 | 0.1775 |
| qwen25_0p5b | B6 | 42 | mmlu-redux | 5615 | 0.4408 | 0.2448 | 0.1957 |
| qwen25_0p5b | B6 | 42 | openbookqa | 500 | 0.5300 | 0.3576 | 0.2071 |
| qwen25_0p5b | B6 | 43 | ai2-arc | 1150 | 0.5548 | 0.3648 | 0.1623 |
| qwen25_0p5b | B6 | 43 | mmlu-redux | 5615 | 0.4362 | 0.2223 | 0.1673 |
| qwen25_0p5b | B6 | 43 | openbookqa | 500 | 0.4940 | 0.2914 | 0.1970 |
| qwen25_0p5b | B6 | 44 | ai2-arc | 1150 | 0.4400 | 0.1453 | 0.1212 |
| qwen25_0p5b | B6 | 44 | mmlu-redux | 5615 | 0.3731 | 0.0921 | 0.1057 |
| qwen25_0p5b | B6 | 44 | openbookqa | 500 | 0.4340 | 0.1325 | 0.1061 |
| qwen3_1p7b | B0 | 42 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 42 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 42 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 43 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 43 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 43 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 44 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 44 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| qwen3_1p7b | B0 | 44 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| qwen3_1p7b | B1 | 42 | ai2-arc | 1150 | 0.5670 | 0.3299 | 0.0801 |
| qwen3_1p7b | B1 | 42 | mmlu-redux | 5615 | 0.4321 | 0.2029 | 0.1429 |
| qwen3_1p7b | B1 | 42 | openbookqa | 500 | 0.5520 | 0.3079 | 0.0758 |
| qwen3_1p7b | B1 | 43 | ai2-arc | 1150 | 0.5400 | 0.3852 | 0.2294 |
| qwen3_1p7b | B1 | 43 | mmlu-redux | 5615 | 0.4534 | 0.3235 | 0.3055 |
| qwen3_1p7b | B1 | 43 | openbookqa | 500 | 0.4920 | 0.3411 | 0.2778 |
| qwen3_1p7b | B1 | 44 | ai2-arc | 1150 | 0.6174 | 0.4404 | 0.1190 |
| qwen3_1p7b | B1 | 44 | mmlu-redux | 5615 | 0.4588 | 0.2637 | 0.1795 |
| qwen3_1p7b | B1 | 44 | openbookqa | 500 | 0.5340 | 0.3444 | 0.1768 |
| qwen3_1p7b | B2 | 42 | ai2-arc | 1150 | 0.6070 | 0.4375 | 0.1407 |
| qwen3_1p7b | B2 | 42 | mmlu-redux | 5615 | 0.4800 | 0.2947 | 0.1764 |
| qwen3_1p7b | B2 | 42 | openbookqa | 500 | 0.5600 | 0.3775 | 0.1616 |
| qwen3_1p7b | B2 | 43 | ai2-arc | 1150 | 0.5278 | 0.3154 | 0.1558 |
| qwen3_1p7b | B2 | 43 | mmlu-redux | 5615 | 0.4383 | 0.2281 | 0.1718 |
| qwen3_1p7b | B2 | 43 | openbookqa | 500 | 0.4900 | 0.2682 | 0.1717 |
| qwen3_1p7b | B2 | 44 | ai2-arc | 1150 | 0.5739 | 0.3648 | 0.1147 |
| qwen3_1p7b | B2 | 44 | mmlu-redux | 5615 | 0.4344 | 0.2072 | 0.1444 |
| qwen3_1p7b | B2 | 44 | openbookqa | 500 | 0.5420 | 0.3344 | 0.1414 |
| qwen3_1p7b | B3 | 42 | ai2-arc | 1150 | 0.6043 | 0.4302 | 0.1364 |
| qwen3_1p7b | B3 | 42 | mmlu-redux | 5615 | 0.4846 | 0.2988 | 0.1708 |
| qwen3_1p7b | B3 | 42 | openbookqa | 500 | 0.5700 | 0.3874 | 0.1515 |
| qwen3_1p7b | B3 | 43 | ai2-arc | 1150 | 0.6391 | 0.5160 | 0.1775 |
| qwen3_1p7b | B3 | 43 | mmlu-redux | 5615 | 0.4953 | 0.3780 | 0.2872 |
| qwen3_1p7b | B3 | 43 | openbookqa | 500 | 0.5660 | 0.4139 | 0.2020 |
| qwen3_1p7b | B3 | 44 | ai2-arc | 1150 | 0.5939 | 0.4012 | 0.1190 |
| qwen3_1p7b | B3 | 44 | mmlu-redux | 5615 | 0.4597 | 0.2536 | 0.1581 |
| qwen3_1p7b | B3 | 44 | openbookqa | 500 | 0.5420 | 0.3510 | 0.1667 |
| qwen3_1p7b | B5 | 42 | ai2-arc | 1150 | 0.5661 | 0.3735 | 0.1472 |
| qwen3_1p7b | B5 | 42 | mmlu-redux | 5615 | 0.4579 | 0.2508 | 0.1581 |
| qwen3_1p7b | B5 | 42 | openbookqa | 500 | 0.5060 | 0.2781 | 0.1465 |
| qwen3_1p7b | B5 | 43 | ai2-arc | 1150 | 0.5774 | 0.4172 | 0.1840 |
| qwen3_1p7b | B5 | 43 | mmlu-redux | 5615 | 0.4620 | 0.3114 | 0.2588 |
| qwen3_1p7b | B5 | 43 | openbookqa | 500 | 0.5320 | 0.4040 | 0.2727 |
| qwen3_1p7b | B5 | 44 | ai2-arc | 1150 | 0.5061 | 0.2238 | 0.0736 |
| qwen3_1p7b | B5 | 44 | mmlu-redux | 5615 | 0.4290 | 0.1708 | 0.0920 |
| qwen3_1p7b | B5 | 44 | openbookqa | 500 | 0.4760 | 0.1788 | 0.0707 |
| qwen3_1p7b | B6 | 42 | ai2-arc | 1150 | 0.6200 | 0.4578 | 0.1385 |
| qwen3_1p7b | B6 | 42 | mmlu-redux | 5615 | 0.4791 | 0.3007 | 0.1901 |
| qwen3_1p7b | B6 | 42 | openbookqa | 500 | 0.5480 | 0.3477 | 0.1465 |
| qwen3_1p7b | B6 | 43 | ai2-arc | 1150 | 0.6009 | 0.4564 | 0.1840 |
| qwen3_1p7b | B6 | 43 | mmlu-redux | 5615 | 0.4693 | 0.3081 | 0.2318 |
| qwen3_1p7b | B6 | 43 | openbookqa | 500 | 0.5300 | 0.3808 | 0.2424 |
| qwen3_1p7b | B6 | 44 | ai2-arc | 1150 | 0.5991 | 0.4317 | 0.1515 |
| qwen3_1p7b | B6 | 44 | mmlu-redux | 5615 | 0.4853 | 0.3287 | 0.2242 |
| qwen3_1p7b | B6 | 44 | openbookqa | 500 | 0.5500 | 0.3642 | 0.1667 |
| tinyllama | B0 | 42 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| tinyllama | B0 | 42 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| tinyllama | B0 | 42 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| tinyllama | B0 | 43 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| tinyllama | B0 | 43 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| tinyllama | B0 | 43 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| tinyllama | B0 | 44 | ai2-arc | 1150 | 0.4017 | 0.0000 | 0.0000 |
| tinyllama | B0 | 44 | mmlu-redux | 5615 | 0.3503 | 0.0000 | 0.0000 |
| tinyllama | B0 | 44 | openbookqa | 500 | 0.3960 | 0.0000 | 0.0000 |
| tinyllama | B1 | 42 | ai2-arc | 1150 | 0.4957 | 0.3823 | 0.3355 |
| tinyllama | B1 | 42 | mmlu-redux | 5615 | 0.4356 | 0.3385 | 0.3843 |
| tinyllama | B1 | 42 | openbookqa | 500 | 0.4420 | 0.3510 | 0.4192 |
| tinyllama | B2 | 42 | ai2-arc | 1150 | 0.5417 | 0.3387 | 0.1558 |
| tinyllama | B2 | 42 | mmlu-redux | 5615 | 0.4363 | 0.2182 | 0.1591 |
| tinyllama | B2 | 42 | openbookqa | 500 | 0.5020 | 0.2781 | 0.1566 |
| tinyllama | B2 | 43 | ai2-arc | 1150 | 0.4591 | 0.2253 | 0.1926 |
| tinyllama | B2 | 43 | mmlu-redux | 5615 | 0.3943 | 0.1976 | 0.2410 |
| tinyllama | B2 | 43 | openbookqa | 500 | 0.4140 | 0.1556 | 0.1919 |
| tinyllama | B2 | 44 | ai2-arc | 1150 | 0.4783 | 0.2267 | 0.1472 |
| tinyllama | B2 | 44 | mmlu-redux | 5615 | 0.3902 | 0.1321 | 0.1312 |
| tinyllama | B2 | 44 | openbookqa | 500 | 0.4640 | 0.1887 | 0.1162 |
| tinyllama | B2-constant | 42 | ai2-arc | 1150 | 0.4957 | 0.2762 | 0.1775 |
| tinyllama | B2-constant | 42 | mmlu-redux | 5615 | 0.4312 | 0.2190 | 0.1754 |
| tinyllama | B2-constant | 42 | openbookqa | 500 | 0.4620 | 0.2417 | 0.2020 |
| tinyllama | B2-constant | 43 | ai2-arc | 1150 | 0.5252 | 0.2980 | 0.1364 |
| tinyllama | B2-constant | 43 | mmlu-redux | 5615 | 0.4262 | 0.2042 | 0.1622 |
| tinyllama | B2-constant | 43 | openbookqa | 500 | 0.4540 | 0.2219 | 0.1919 |
| tinyllama | B2-constant | 44 | ai2-arc | 1150 | 0.4948 | 0.2049 | 0.0736 |
| tinyllama | B2-constant | 44 | mmlu-redux | 5615 | 0.4077 | 0.1436 | 0.1027 |
| tinyllama | B2-constant | 44 | openbookqa | 500 | 0.4440 | 0.1258 | 0.0707 |
| tinyllama | B3 | 42 | ai2-arc | 1150 | 0.4704 | 0.1904 | 0.1126 |
| tinyllama | B3 | 42 | mmlu-redux | 5615 | 0.3932 | 0.1349 | 0.1276 |
| tinyllama | B3 | 42 | openbookqa | 500 | 0.4320 | 0.1391 | 0.1212 |
| tinyllama | B3 | 43 | ai2-arc | 1150 | 0.5157 | 0.3735 | 0.2727 |
| tinyllama | B3 | 43 | mmlu-redux | 5615 | 0.4397 | 0.3139 | 0.3269 |
| tinyllama | B3 | 43 | openbookqa | 500 | 0.4560 | 0.3113 | 0.3232 |
| tinyllama | B3 | 44 | ai2-arc | 1150 | 0.5365 | 0.3648 | 0.2078 |
| tinyllama | B3 | 44 | mmlu-redux | 5615 | 0.4591 | 0.3076 | 0.2598 |
| tinyllama | B3 | 44 | openbookqa | 500 | 0.5020 | 0.3675 | 0.2929 |
| tinyllama | B4 | 42 | ai2-arc | 1150 | 0.5513 | 0.4070 | 0.2338 |
| tinyllama | B4 | 42 | mmlu-redux | 5615 | 0.4500 | 0.3133 | 0.2964 |
| tinyllama | B4 | 42 | openbookqa | 500 | 0.5020 | 0.3411 | 0.2525 |
| tinyllama | B4 | 43 | ai2-arc | 1150 | 0.5200 | 0.3634 | 0.2468 |
| tinyllama | B4 | 43 | mmlu-redux | 5615 | 0.4454 | 0.3098 | 0.3030 |
| tinyllama | B4 | 43 | openbookqa | 500 | 0.4640 | 0.3278 | 0.3283 |
| tinyllama | B4 | 44 | ai2-arc | 1150 | 0.5130 | 0.2892 | 0.1537 |
| tinyllama | B4 | 44 | mmlu-redux | 5615 | 0.4285 | 0.2163 | 0.1779 |
| tinyllama | B4 | 44 | openbookqa | 500 | 0.4860 | 0.2682 | 0.1818 |
| tinyllama | B5 | 42 | ai2-arc | 1150 | 0.5191 | 0.4070 | 0.3139 |
| tinyllama | B5 | 42 | mmlu-redux | 5615 | 0.4513 | 0.3506 | 0.3620 |
| tinyllama | B5 | 42 | openbookqa | 500 | 0.4180 | 0.3344 | 0.4545 |
| tinyllama | B5 | 43 | ai2-arc | 1150 | 0.4896 | 0.2544 | 0.1602 |
| tinyllama | B5 | 43 | mmlu-redux | 5615 | 0.4004 | 0.1626 | 0.1586 |
| tinyllama | B5 | 43 | openbookqa | 500 | 0.4220 | 0.1225 | 0.1212 |
| tinyllama | B5 | 44 | ai2-arc | 1150 | 0.5313 | 0.3648 | 0.2208 |
| tinyllama | B5 | 44 | mmlu-redux | 5615 | 0.4488 | 0.2760 | 0.2308 |
| tinyllama | B5 | 44 | openbookqa | 500 | 0.4700 | 0.3113 | 0.2879 |
| tinyllama | B6 | 42 | ai2-arc | 1150 | 0.5478 | 0.4099 | 0.2468 |
| tinyllama | B6 | 42 | mmlu-redux | 5615 | 0.4703 | 0.3520 | 0.3101 |
| tinyllama | B6 | 42 | openbookqa | 500 | 0.5060 | 0.3510 | 0.2576 |
| tinyllama | B6 | 43 | ai2-arc | 1150 | 0.5296 | 0.3372 | 0.1840 |
| tinyllama | B6 | 43 | mmlu-redux | 5615 | 0.4463 | 0.2547 | 0.1983 |
| tinyllama | B6 | 43 | openbookqa | 500 | 0.4700 | 0.2682 | 0.2222 |
| tinyllama | B6 | 44 | ai2-arc | 1150 | 0.5313 | 0.3997 | 0.2727 |
| tinyllama | B6 | 44 | mmlu-redux | 5615 | 0.4543 | 0.3314 | 0.3177 |
| tinyllama | B6 | 44 | openbookqa | 500 | 0.4740 | 0.3974 | 0.4091 |
| tinyllama | B6-constant | 42 | ai2-arc | 1150 | 0.5496 | 0.4099 | 0.2424 |
| tinyllama | B6-constant | 42 | mmlu-redux | 5615 | 0.4559 | 0.3174 | 0.2872 |
| tinyllama | B6-constant | 42 | openbookqa | 500 | 0.5280 | 0.4007 | 0.2778 |
| tinyllama | B6-shuffle | 42 | ai2-arc | 1150 | 0.5226 | 0.4273 | 0.3355 |
| tinyllama | B6-shuffle | 42 | mmlu-redux | 5615 | 0.4490 | 0.3695 | 0.4037 |
| tinyllama | B6-shuffle | 42 | openbookqa | 500 | 0.4720 | 0.4172 | 0.4444 |

Positive transfer is conditioned on receiver-wrong examples; negative transfer is conditioned on receiver-correct examples.

## Macro and sample-weighted means

| pair | method | seed | tasks | macro mean | weighted mean |
| --- | --- | --- | --- | --- | --- |
| llama32_1b | B0 | 42 | 3 | 0.3827 | 0.3616 |
| llama32_1b | B0 | 43 | 3 | 0.3827 | 0.3616 |
| llama32_1b | B0 | 44 | 3 | 0.3827 | 0.3616 |
| llama32_1b | B1 | 42 | 3 | 0.4575 | 0.4431 |
| llama32_1b | B1 | 43 | 3 | 0.4893 | 0.4690 |
| llama32_1b | B1 | 44 | 3 | 0.4946 | 0.4632 |
| llama32_1b | B2 | 42 | 3 | 0.4910 | 0.4639 |
| llama32_1b | B2 | 43 | 3 | 0.5068 | 0.4732 |
| llama32_1b | B2 | 44 | 3 | 0.4883 | 0.4586 |
| llama32_1b | B3 | 42 | 3 | 0.4910 | 0.4640 |
| llama32_1b | B3 | 43 | 3 | 0.5069 | 0.4735 |
| llama32_1b | B3 | 44 | 3 | 0.4883 | 0.4586 |
| llama32_1b | B5 | 42 | 3 | 0.4831 | 0.4691 |
| llama32_1b | B5 | 43 | 3 | 0.4969 | 0.4665 |
| llama32_1b | B5 | 44 | 3 | 0.4828 | 0.4652 |
| llama32_1b | B6 | 42 | 3 | 0.4937 | 0.4640 |
| llama32_1b | B6 | 43 | 3 | 0.5078 | 0.4778 |
| llama32_1b | B6 | 44 | 3 | 0.5003 | 0.4591 |
| qwen25_0p5b | B0 | 42 | 3 | 0.3827 | 0.3616 |
| qwen25_0p5b | B0 | 43 | 3 | 0.3827 | 0.3616 |
| qwen25_0p5b | B0 | 44 | 3 | 0.3827 | 0.3616 |
| qwen25_0p5b | B1 | 42 | 3 | 0.4965 | 0.4723 |
| qwen25_0p5b | B1 | 43 | 3 | 0.4555 | 0.4398 |
| qwen25_0p5b | B1 | 44 | 3 | 0.4207 | 0.4100 |
| qwen25_0p5b | B2 | 42 | 3 | 0.4905 | 0.4603 |
| qwen25_0p5b | B2 | 43 | 3 | 0.4641 | 0.4427 |
| qwen25_0p5b | B2 | 44 | 3 | 0.4776 | 0.4504 |
| qwen25_0p5b | B3 | 42 | 3 | 0.4838 | 0.4489 |
| qwen25_0p5b | B3 | 43 | 3 | 0.4804 | 0.4591 |
| qwen25_0p5b | B3 | 44 | 3 | 0.4673 | 0.4434 |
| qwen25_0p5b | B5 | 42 | 3 | 0.4725 | 0.4427 |
| qwen25_0p5b | B5 | 43 | 3 | 0.5013 | 0.4716 |
| qwen25_0p5b | B5 | 44 | 3 | 0.4576 | 0.4275 |
| qwen25_0p5b | B6 | 42 | 3 | 0.5021 | 0.4619 |
| qwen25_0p5b | B6 | 43 | 3 | 0.4950 | 0.4589 |
| qwen25_0p5b | B6 | 44 | 3 | 0.4157 | 0.3879 |
| qwen3_1p7b | B0 | 42 | 3 | 0.3827 | 0.3616 |
| qwen3_1p7b | B0 | 43 | 3 | 0.3827 | 0.3616 |
| qwen3_1p7b | B0 | 44 | 3 | 0.3827 | 0.3616 |
| qwen3_1p7b | B1 | 42 | 3 | 0.5170 | 0.4617 |
| qwen3_1p7b | B1 | 43 | 3 | 0.4951 | 0.4698 |
| qwen3_1p7b | B1 | 44 | 3 | 0.5367 | 0.4891 |
| qwen3_1p7b | B2 | 42 | 3 | 0.5490 | 0.5056 |
| qwen3_1p7b | B2 | 43 | 3 | 0.4854 | 0.4560 |
| qwen3_1p7b | B2 | 44 | 3 | 0.5168 | 0.4639 |
| qwen3_1p7b | B3 | 42 | 3 | 0.5530 | 0.5094 |
| qwen3_1p7b | B3 | 43 | 3 | 0.5668 | 0.5229 |
| qwen3_1p7b | B3 | 44 | 3 | 0.5319 | 0.4866 |
| qwen3_1p7b | B5 | 42 | 3 | 0.5100 | 0.4783 |
| qwen3_1p7b | B5 | 43 | 3 | 0.5238 | 0.4851 |
| qwen3_1p7b | B5 | 44 | 3 | 0.4704 | 0.4445 |
| qwen3_1p7b | B6 | 42 | 3 | 0.5490 | 0.5061 |
| qwen3_1p7b | B6 | 43 | 3 | 0.5334 | 0.4943 |
| qwen3_1p7b | B6 | 44 | 3 | 0.5448 | 0.5078 |
| tinyllama | B0 | 42 | 3 | 0.3827 | 0.3616 |
| tinyllama | B0 | 43 | 3 | 0.3827 | 0.3616 |
| tinyllama | B0 | 44 | 3 | 0.3827 | 0.3616 |
| tinyllama | B1 | 42 | 3 | 0.4578 | 0.4456 |
| tinyllama | B2 | 42 | 3 | 0.4934 | 0.4575 |
| tinyllama | B2 | 43 | 3 | 0.4225 | 0.4059 |
| tinyllama | B2 | 44 | 3 | 0.4442 | 0.4092 |
| tinyllama | B2-constant | 42 | 3 | 0.4629 | 0.4435 |
| tinyllama | B2-constant | 43 | 3 | 0.4685 | 0.4438 |
| tinyllama | B2-constant | 44 | 3 | 0.4488 | 0.4240 |
| tinyllama | B3 | 42 | 3 | 0.4319 | 0.4081 |
| tinyllama | B3 | 43 | 3 | 0.4705 | 0.4529 |
| tinyllama | B3 | 44 | 3 | 0.4992 | 0.4743 |
| tinyllama | B4 | 42 | 3 | 0.5011 | 0.4696 |
| tinyllama | B4 | 43 | 3 | 0.4765 | 0.4585 |
| tinyllama | B4 | 44 | 3 | 0.4758 | 0.4458 |
| tinyllama | B5 | 42 | 3 | 0.4628 | 0.4597 |
| tinyllama | B5 | 43 | 3 | 0.4373 | 0.4160 |
| tinyllama | B5 | 44 | 3 | 0.4834 | 0.4633 |
| tinyllama | B6 | 42 | 3 | 0.5081 | 0.4851 |
| tinyllama | B6 | 43 | 3 | 0.4820 | 0.4611 |
| tinyllama | B6 | 44 | 3 | 0.4865 | 0.4679 |
| tinyllama | B6-constant | 42 | 3 | 0.5112 | 0.4757 |
| tinyllama | B6-shuffle | 42 | 3 | 0.4812 | 0.4622 |

## Component contributions

| pair | component contrast | candidate - baseline | seeds | sample std | CI low | CI high | bootstrap level | positive seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| __all__ | c2c_longest_vs_receiver | 0.0929 | 3 | missing | 0.0768 | 0.1077 | pair_cluster_across_pairs | None |
| __all__ | entropy_position | 0.0228 | 1 | missing | 0.0139 | 0.0318 | pair_cluster_across_pairs | None |
| __all__ | entropy_values | 0.0094 | 1 | missing | 0.0011 | 0.0175 | pair_cluster_across_pairs | None |
| __all__ | full_over_gate_only | 0.0119 | 3 | missing | -0.0092 | 0.0331 | pair_cluster_across_pairs | None |
| __all__ | full_over_hard_span | 0.0154 | 3 | missing | -0.0114 | 0.0405 | pair_cluster_across_pairs | None |
| __all__ | full_over_static_entropy | 0.0134 | 3 | missing | 0.0028 | 0.0232 | pair_cluster_across_pairs | None |
| __all__ | gate_capacity | 0.0093 | 3 | missing | -0.0266 | 0.0390 | pair_cluster_across_pairs | None |
| __all__ | gate_capacity_confounded | 0.0035 | 3 | missing | -0.0115 | 0.0213 | pair_cluster_across_pairs | None |
| __all__ | hard_span_vs_longest | 0.0077 | 3 | missing | -0.0055 | 0.0198 | pair_cluster_across_pairs | None |
| __all__ | hard_span_vs_receiver | 0.0923 | 3 | missing | 0.0696 | 0.1125 | pair_cluster_across_pairs | None |
| __all__ | soft_candidates | 0.0129 | 3 | missing | -0.0061 | 0.0358 | pair_cluster_across_pairs | None |
| __all__ | static_entropy | 0.0129 | 3 | missing | -0.0278 | 0.0598 | pair_cluster_across_pairs | None |
| llama32_1b | c2c_longest_vs_receiver | 0.0968 | 3 | 0.0136 | 0.0819 | 0.1103 | pair_seed_cluster_within_pair | 3 |
| llama32_1b | full_over_gate_only | 0.0000 | 3 | 0.0098 | -0.0093 | 0.0109 | pair_seed_cluster_within_pair | 1 |
| llama32_1b | full_over_hard_span | 0.0017 | 3 | 0.0025 | -0.0042 | 0.0076 | pair_seed_cluster_within_pair | 3 |
| llama32_1b | gate_capacity_confounded | 0.0017 | 3 | 0.0073 | -0.0072 | 0.0092 | pair_seed_cluster_within_pair | 2 |
| llama32_1b | hard_span_vs_longest | 0.0068 | 3 | 0.0129 | -0.0054 | 0.0202 | pair_seed_cluster_within_pair | 2 |
| llama32_1b | hard_span_vs_receiver | 0.1036 | 3 | 0.0074 | 0.0943 | 0.1137 | pair_seed_cluster_within_pair | 3 |
| llama32_1b | soft_candidates | 0.0001 | 3 | 0.0001 | -0.0001 | 0.0004 | pair_seed_cluster_within_pair | 2 |
| qwen25_0p5b | c2c_longest_vs_receiver | 0.0791 | 3 | 0.0311 | 0.0493 | 0.1087 | pair_seed_cluster_within_pair | 3 |
| qwen25_0p5b | full_over_gate_only | -0.0110 | 3 | 0.0295 | -0.0386 | 0.0174 | pair_seed_cluster_within_pair | 1 |
| qwen25_0p5b | full_over_hard_span | -0.0149 | 3 | 0.0419 | -0.0612 | 0.0161 | pair_seed_cluster_within_pair | 2 |
| qwen25_0p5b | gate_capacity_confounded | -0.0039 | 3 | 0.0285 | -0.0245 | 0.0279 | pair_seed_cluster_within_pair | 1 |
| qwen25_0p5b | hard_span_vs_longest | 0.0104 | 3 | 0.0270 | -0.0120 | 0.0387 | pair_seed_cluster_within_pair | 2 |
| qwen25_0p5b | hard_span_vs_receiver | 0.0895 | 3 | 0.0088 | 0.0797 | 0.0999 | pair_seed_cluster_within_pair | 3 |
| qwen25_0p5b | soft_candidates | -0.0007 | 3 | 0.0149 | -0.0129 | 0.0149 | pair_seed_cluster_within_pair | 1 |
| qwen3_1p7b | c2c_longest_vs_receiver | 0.1119 | 3 | 0.0141 | 0.0984 | 0.1271 | pair_seed_cluster_within_pair | 3 |
| qwen3_1p7b | full_over_gate_only | 0.0334 | 3 | 0.0275 | 0.0103 | 0.0629 | pair_seed_cluster_within_pair | 3 |
| qwen3_1p7b | full_over_hard_span | 0.0276 | 3 | 0.0236 | 0.0022 | 0.0459 | pair_seed_cluster_within_pair | 3 |
| qwen3_1p7b | gate_capacity_confounded | -0.0059 | 3 | 0.0305 | -0.0280 | 0.0280 | pair_seed_cluster_within_pair | 1 |
| qwen3_1p7b | hard_span_vs_longest | 0.0017 | 3 | 0.0370 | -0.0251 | 0.0418 | pair_seed_cluster_within_pair | 1 |
| qwen3_1p7b | hard_span_vs_receiver | 0.1136 | 3 | 0.0266 | 0.0934 | 0.1435 | pair_seed_cluster_within_pair | 3 |
| qwen3_1p7b | soft_candidates | 0.0312 | 3 | 0.0324 | 0.0047 | 0.0659 | pair_seed_cluster_within_pair | 3 |
| tinyllama | c2c_longest_vs_receiver | 0.0840 | 1 | missing | 0.0701 | 0.0976 | pair_seed_cluster_within_pair | 1 |
| tinyllama | entropy_position | 0.0228 | 1 | missing | 0.0138 | 0.0317 | pair_seed_cluster_within_pair | 1 |
| tinyllama | entropy_values | 0.0094 | 1 | missing | 0.0015 | 0.0173 | pair_seed_cluster_within_pair | 1 |
| tinyllama | full_over_gate_only | 0.0250 | 3 | 0.0203 | 0.0053 | 0.0440 | pair_seed_cluster_within_pair | 3 |
| tinyllama | full_over_hard_span | 0.0471 | 3 | 0.0171 | 0.0283 | 0.0617 | pair_seed_cluster_within_pair | 3 |
| tinyllama | full_over_static_entropy | 0.0134 | 3 | 0.0099 | 0.0026 | 0.0234 | pair_seed_cluster_within_pair | 3 |
| tinyllama | gate_capacity | 0.0093 | 3 | 0.0341 | -0.0260 | 0.0381 | pair_seed_cluster_within_pair | 2 |
| tinyllama | gate_capacity_confounded | 0.0221 | 3 | 0.0280 | 0.0006 | 0.0526 | pair_seed_cluster_within_pair | 3 |
| tinyllama | hard_span_vs_longest | 0.0120 | 1 | missing | 0.0004 | 0.0233 | pair_seed_cluster_within_pair | 1 |
| tinyllama | hard_span_vs_receiver | 0.0626 | 3 | 0.0289 | 0.0417 | 0.0937 | pair_seed_cluster_within_pair | 3 |
| tinyllama | soft_candidates | 0.0209 | 3 | 0.0615 | -0.0485 | 0.0650 | pair_seed_cluster_within_pair | 2 |
| tinyllama | static_entropy | 0.0129 | 3 | 0.0454 | -0.0279 | 0.0596 | pair_seed_cluster_within_pair | 2 |

## Paired mechanism gains in fixed candidate buckets

| pair | contrast | seed | candidate-defined field | bucket | n | delta | CI low | CI high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama32_1b | full_over_gate_only | 42 | alignment_bucket | 1-to-1 | 7168 | -0.0039 | -0.0130 | 0.0050 |
| llama32_1b | full_over_gate_only | 42 | alignment_bucket | one-to-many | 97 | -0.0928 | -0.1856 | 0.0000 |
| llama32_1b | full_over_gate_only | 42 | alignment_entropy | (0,0.25] | 97 | -0.0928 | -0.1856 | 0.0000 |
| llama32_1b | full_over_gate_only | 42 | alignment_entropy | 0 | 7168 | -0.0039 | -0.0131 | 0.0052 |
| llama32_1b | full_over_gate_only | 42 | boundary_mismatch | 0 | 7265 | -0.0051 | -0.0139 | 0.0039 |
| llama32_1b | full_over_gate_only | 42 | candidate_count | 1 | 7265 | -0.0051 | -0.0143 | 0.0043 |
| llama32_1b | full_over_gate_only | 43 | alignment_bucket | 1-to-1 | 7168 | 0.0116 | 0.0036 | 0.0194 |
| llama32_1b | full_over_gate_only | 43 | alignment_bucket | one-to-many | 97 | -0.0103 | -0.0928 | 0.0722 |
| llama32_1b | full_over_gate_only | 43 | alignment_entropy | (0,0.25] | 97 | -0.0103 | -0.0928 | 0.0722 |
| llama32_1b | full_over_gate_only | 43 | alignment_entropy | 0 | 7168 | 0.0116 | 0.0033 | 0.0197 |
| llama32_1b | full_over_gate_only | 43 | boundary_mismatch | 0 | 7265 | 0.0113 | 0.0036 | 0.0187 |
| llama32_1b | full_over_gate_only | 43 | candidate_count | 1 | 7265 | 0.0113 | 0.0036 | 0.0195 |
| llama32_1b | full_over_gate_only | 44 | alignment_bucket | 1-to-1 | 7168 | -0.0061 | -0.0151 | 0.0032 |
| llama32_1b | full_over_gate_only | 44 | alignment_bucket | one-to-many | 97 | -0.0103 | -0.0928 | 0.0722 |
| llama32_1b | full_over_gate_only | 44 | alignment_entropy | (0,0.25] | 97 | -0.0103 | -0.0825 | 0.0619 |
| llama32_1b | full_over_gate_only | 44 | alignment_entropy | 0 | 7168 | -0.0061 | -0.0153 | 0.0032 |
| llama32_1b | full_over_gate_only | 44 | boundary_mismatch | 0 | 7265 | -0.0062 | -0.0150 | 0.0026 |
| llama32_1b | full_over_gate_only | 44 | candidate_count | 1 | 7265 | -0.0062 | -0.0154 | 0.0033 |
| llama32_1b | gate_capacity_confounded | 42 | alignment_bucket | 1-to-1 | 7265 | 0.0052 | -0.0026 | 0.0135 |
| llama32_1b | gate_capacity_confounded | 42 | alignment_entropy | 0 | 7265 | 0.0052 | -0.0028 | 0.0134 |
| llama32_1b | gate_capacity_confounded | 42 | boundary_mismatch | 0 | 7265 | 0.0052 | -0.0029 | 0.0135 |
| llama32_1b | gate_capacity_confounded | 42 | candidate_count | 1 | 7265 | 0.0052 | -0.0032 | 0.0135 |
| llama32_1b | gate_capacity_confounded | 43 | alignment_bucket | 1-to-1 | 7265 | -0.0067 | -0.0160 | 0.0023 |
| llama32_1b | gate_capacity_confounded | 43 | alignment_entropy | 0 | 7265 | -0.0067 | -0.0158 | 0.0025 |
| llama32_1b | gate_capacity_confounded | 43 | boundary_mismatch | 0 | 7265 | -0.0067 | -0.0164 | 0.0025 |
| llama32_1b | gate_capacity_confounded | 43 | candidate_count | 1 | 7265 | -0.0067 | -0.0161 | 0.0026 |
| llama32_1b | gate_capacity_confounded | 44 | alignment_bucket | 1-to-1 | 7265 | 0.0066 | -0.0025 | 0.0153 |
| llama32_1b | gate_capacity_confounded | 44 | alignment_entropy | 0 | 7265 | 0.0066 | -0.0021 | 0.0154 |
| llama32_1b | gate_capacity_confounded | 44 | boundary_mismatch | 0 | 7265 | 0.0066 | -0.0023 | 0.0154 |
| llama32_1b | gate_capacity_confounded | 44 | candidate_count | 1 | 7265 | 0.0066 | -0.0023 | 0.0151 |
| llama32_1b | soft_candidates | 42 | alignment_bucket | 1-to-1 | 7168 | 0.0000 | 0.0000 | 0.0000 |
| llama32_1b | soft_candidates | 42 | alignment_bucket | one-to-many | 97 | 0.0103 | 0.0000 | 0.0309 |
| llama32_1b | soft_candidates | 42 | alignment_entropy | (0,0.25] | 97 | 0.0103 | 0.0000 | 0.0309 |
| llama32_1b | soft_candidates | 42 | alignment_entropy | 0 | 7168 | 0.0000 | 0.0000 | 0.0000 |
| llama32_1b | soft_candidates | 42 | boundary_mismatch | 0 | 7265 | 0.0001 | 0.0000 | 0.0004 |
| llama32_1b | soft_candidates | 42 | candidate_count | 1 | 7265 | 0.0001 | 0.0000 | 0.0004 |
| llama32_1b | soft_candidates | 43 | alignment_bucket | 1-to-1 | 7168 | 0.0000 | 0.0000 | 0.0000 |
| llama32_1b | soft_candidates | 43 | alignment_bucket | one-to-many | 97 | 0.0206 | 0.0000 | 0.0515 |
| llama32_1b | soft_candidates | 43 | alignment_entropy | (0,0.25] | 97 | 0.0206 | 0.0000 | 0.0515 |
| llama32_1b | soft_candidates | 43 | alignment_entropy | 0 | 7168 | 0.0000 | 0.0000 | 0.0000 |
| llama32_1b | soft_candidates | 43 | boundary_mismatch | 0 | 7265 | 0.0003 | 0.0000 | 0.0007 |
| llama32_1b | soft_candidates | 43 | candidate_count | 1 | 7265 | 0.0003 | 0.0000 | 0.0007 |
| llama32_1b | soft_candidates | 44 | alignment_bucket | 1-to-1 | 7168 | 0.0000 | 0.0000 | 0.0000 |
| llama32_1b | soft_candidates | 44 | alignment_bucket | one-to-many | 97 | 0.0000 | -0.0309 | 0.0309 |
| llama32_1b | soft_candidates | 44 | alignment_entropy | (0,0.25] | 97 | 0.0000 | -0.0309 | 0.0309 |
| llama32_1b | soft_candidates | 44 | alignment_entropy | 0 | 7168 | 0.0000 | 0.0000 | 0.0000 |
| llama32_1b | soft_candidates | 44 | boundary_mismatch | 0 | 7265 | 0.0000 | -0.0004 | 0.0004 |
| llama32_1b | soft_candidates | 44 | candidate_count | 1 | 7265 | 0.0000 | -0.0004 | 0.0004 |
| qwen25_0p5b | full_over_gate_only | 42 | alignment_bucket | 1-to-1 | 7161 | 0.0198 | 0.0105 | 0.0296 |
| qwen25_0p5b | full_over_gate_only | 42 | alignment_bucket | one-to-many | 104 | -0.0192 | -0.1154 | 0.0769 |
| qwen25_0p5b | full_over_gate_only | 42 | alignment_entropy | (0,0.25] | 104 | -0.0192 | -0.1154 | 0.0769 |
| qwen25_0p5b | full_over_gate_only | 42 | alignment_entropy | 0 | 7161 | 0.0198 | 0.0101 | 0.0292 |
| qwen25_0p5b | full_over_gate_only | 42 | boundary_mismatch | 0 | 7265 | 0.0193 | 0.0099 | 0.0288 |
| qwen25_0p5b | full_over_gate_only | 42 | candidate_count | 1 | 7265 | 0.0193 | 0.0099 | 0.0289 |
| qwen25_0p5b | full_over_gate_only | 43 | alignment_bucket | 1-to-1 | 7161 | -0.0124 | -0.0201 | -0.0049 |
| qwen25_0p5b | full_over_gate_only | 43 | alignment_bucket | one-to-many | 104 | -0.0288 | -0.0962 | 0.0385 |
| qwen25_0p5b | full_over_gate_only | 43 | alignment_entropy | (0,0.25] | 104 | -0.0288 | -0.0962 | 0.0385 |
| qwen25_0p5b | full_over_gate_only | 43 | alignment_entropy | 0 | 7161 | -0.0124 | -0.0204 | -0.0049 |
| qwen25_0p5b | full_over_gate_only | 43 | boundary_mismatch | 0 | 7265 | -0.0127 | -0.0204 | -0.0048 |
| qwen25_0p5b | full_over_gate_only | 43 | candidate_count | 1 | 7265 | -0.0127 | -0.0204 | -0.0051 |
| qwen25_0p5b | full_over_gate_only | 44 | alignment_bucket | 1-to-1 | 7161 | -0.0401 | -0.0482 | -0.0324 |
| qwen25_0p5b | full_over_gate_only | 44 | alignment_bucket | one-to-many | 104 | -0.0096 | -0.0577 | 0.0385 |
| qwen25_0p5b | full_over_gate_only | 44 | alignment_entropy | (0,0.25] | 104 | -0.0096 | -0.0577 | 0.0385 |
| qwen25_0p5b | full_over_gate_only | 44 | alignment_entropy | 0 | 7161 | -0.0401 | -0.0479 | -0.0323 |
| qwen25_0p5b | full_over_gate_only | 44 | boundary_mismatch | 0 | 7265 | -0.0396 | -0.0474 | -0.0318 |
| qwen25_0p5b | full_over_gate_only | 44 | candidate_count | 1 | 7265 | -0.0396 | -0.0472 | -0.0318 |
| qwen25_0p5b | gate_capacity_confounded | 42 | alignment_bucket | 1-to-1 | 7265 | -0.0176 | -0.0270 | -0.0083 |
| qwen25_0p5b | gate_capacity_confounded | 42 | alignment_entropy | 0 | 7265 | -0.0176 | -0.0270 | -0.0084 |
| qwen25_0p5b | gate_capacity_confounded | 42 | boundary_mismatch | 0 | 7265 | -0.0176 | -0.0271 | -0.0081 |
| qwen25_0p5b | gate_capacity_confounded | 42 | candidate_count | 1 | 7265 | -0.0176 | -0.0267 | -0.0083 |
| qwen25_0p5b | gate_capacity_confounded | 43 | alignment_bucket | 1-to-1 | 7265 | 0.0289 | 0.0197 | 0.0380 |
| qwen25_0p5b | gate_capacity_confounded | 43 | alignment_entropy | 0 | 7265 | 0.0289 | 0.0197 | 0.0380 |
| qwen25_0p5b | gate_capacity_confounded | 43 | boundary_mismatch | 0 | 7265 | 0.0289 | 0.0194 | 0.0379 |
| qwen25_0p5b | gate_capacity_confounded | 43 | candidate_count | 1 | 7265 | 0.0289 | 0.0198 | 0.0383 |
| qwen25_0p5b | gate_capacity_confounded | 44 | alignment_bucket | 1-to-1 | 7265 | -0.0228 | -0.0307 | -0.0151 |
| qwen25_0p5b | gate_capacity_confounded | 44 | alignment_entropy | 0 | 7265 | -0.0228 | -0.0306 | -0.0150 |
| qwen25_0p5b | gate_capacity_confounded | 44 | boundary_mismatch | 0 | 7265 | -0.0228 | -0.0308 | -0.0149 |
| qwen25_0p5b | gate_capacity_confounded | 44 | candidate_count | 1 | 7265 | -0.0228 | -0.0308 | -0.0150 |
| qwen25_0p5b | soft_candidates | 42 | alignment_bucket | 1-to-1 | 7161 | -0.0117 | -0.0197 | -0.0038 |
| qwen25_0p5b | soft_candidates | 42 | alignment_bucket | one-to-many | 104 | 0.0096 | -0.0673 | 0.0865 |
| qwen25_0p5b | soft_candidates | 42 | alignment_entropy | (0,0.25] | 104 | 0.0096 | -0.0673 | 0.0865 |
| qwen25_0p5b | soft_candidates | 42 | alignment_entropy | 0 | 7161 | -0.0117 | -0.0196 | -0.0038 |
| qwen25_0p5b | soft_candidates | 42 | boundary_mismatch | 0 | 7265 | -0.0114 | -0.0193 | -0.0034 |
| qwen25_0p5b | soft_candidates | 42 | candidate_count | 1 | 7265 | -0.0114 | -0.0193 | -0.0034 |
| qwen25_0p5b | soft_candidates | 43 | alignment_bucket | 1-to-1 | 7161 | 0.0163 | 0.0071 | 0.0260 |
| qwen25_0p5b | soft_candidates | 43 | alignment_bucket | one-to-many | 104 | 0.0192 | -0.0769 | 0.1154 |
| qwen25_0p5b | soft_candidates | 43 | alignment_entropy | (0,0.25] | 104 | 0.0192 | -0.0769 | 0.1154 |
| qwen25_0p5b | soft_candidates | 43 | alignment_entropy | 0 | 7161 | 0.0163 | 0.0067 | 0.0257 |
| qwen25_0p5b | soft_candidates | 43 | boundary_mismatch | 0 | 7265 | 0.0164 | 0.0069 | 0.0256 |
| qwen25_0p5b | soft_candidates | 43 | candidate_count | 1 | 7265 | 0.0164 | 0.0067 | 0.0257 |
| qwen25_0p5b | soft_candidates | 44 | alignment_bucket | 1-to-1 | 7161 | -0.0071 | -0.0137 | -0.0003 |
| qwen25_0p5b | soft_candidates | 44 | alignment_bucket | one-to-many | 104 | 0.0000 | -0.0481 | 0.0481 |
| qwen25_0p5b | soft_candidates | 44 | alignment_entropy | (0,0.25] | 104 | 0.0000 | -0.0481 | 0.0481 |
| qwen25_0p5b | soft_candidates | 44 | alignment_entropy | 0 | 7161 | -0.0071 | -0.0135 | -0.0008 |
| qwen25_0p5b | soft_candidates | 44 | boundary_mismatch | 0 | 7265 | -0.0070 | -0.0136 | -0.0006 |
| qwen25_0p5b | soft_candidates | 44 | candidate_count | 1 | 7265 | -0.0070 | -0.0135 | -0.0004 |
| qwen3_1p7b | full_over_gate_only | 42 | alignment_bucket | 1-to-1 | 7161 | 0.0278 | 0.0201 | 0.0355 |
| qwen3_1p7b | full_over_gate_only | 42 | alignment_bucket | one-to-many | 104 | 0.0288 | -0.0577 | 0.1154 |
| qwen3_1p7b | full_over_gate_only | 42 | alignment_entropy | (0,0.25] | 104 | 0.0288 | -0.0577 | 0.1154 |
| qwen3_1p7b | full_over_gate_only | 42 | alignment_entropy | 0 | 7161 | 0.0278 | 0.0197 | 0.0360 |
| qwen3_1p7b | full_over_gate_only | 42 | boundary_mismatch | 0 | 7265 | 0.0278 | 0.0200 | 0.0357 |
| qwen3_1p7b | full_over_gate_only | 42 | candidate_count | 1 | 7265 | 0.0278 | 0.0201 | 0.0355 |
| qwen3_1p7b | full_over_gate_only | 43 | alignment_bucket | 1-to-1 | 7161 | 0.0102 | 0.0024 | 0.0179 |
| qwen3_1p7b | full_over_gate_only | 43 | alignment_bucket | one-to-many | 104 | -0.0577 | -0.1442 | 0.0288 |
| qwen3_1p7b | full_over_gate_only | 43 | alignment_entropy | (0,0.25] | 104 | -0.0577 | -0.1442 | 0.0288 |
| qwen3_1p7b | full_over_gate_only | 43 | alignment_entropy | 0 | 7161 | 0.0102 | 0.0026 | 0.0179 |
| qwen3_1p7b | full_over_gate_only | 43 | boundary_mismatch | 0 | 7265 | 0.0092 | 0.0019 | 0.0169 |
| qwen3_1p7b | full_over_gate_only | 43 | candidate_count | 1 | 7265 | 0.0092 | 0.0018 | 0.0169 |
| qwen3_1p7b | full_over_gate_only | 44 | alignment_bucket | 1-to-1 | 7161 | 0.0635 | 0.0538 | 0.0732 |
| qwen3_1p7b | full_over_gate_only | 44 | alignment_bucket | one-to-many | 104 | 0.0481 | -0.0481 | 0.1442 |
| qwen3_1p7b | full_over_gate_only | 44 | alignment_entropy | (0,0.25] | 104 | 0.0481 | -0.0481 | 0.1442 |
| qwen3_1p7b | full_over_gate_only | 44 | alignment_entropy | 0 | 7161 | 0.0635 | 0.0542 | 0.0729 |
| qwen3_1p7b | full_over_gate_only | 44 | boundary_mismatch | 0 | 7265 | 0.0633 | 0.0538 | 0.0731 |
| qwen3_1p7b | full_over_gate_only | 44 | candidate_count | 1 | 7265 | 0.0633 | 0.0534 | 0.0727 |
| qwen3_1p7b | gate_capacity_confounded | 42 | alignment_bucket | 1-to-1 | 7265 | -0.0273 | -0.0355 | -0.0187 |
| qwen3_1p7b | gate_capacity_confounded | 42 | alignment_entropy | 0 | 7265 | -0.0273 | -0.0358 | -0.0190 |
| qwen3_1p7b | gate_capacity_confounded | 42 | boundary_mismatch | 0 | 7265 | -0.0273 | -0.0355 | -0.0187 |
| qwen3_1p7b | gate_capacity_confounded | 42 | candidate_count | 1 | 7265 | -0.0273 | -0.0358 | -0.0190 |
| qwen3_1p7b | gate_capacity_confounded | 43 | alignment_bucket | 1-to-1 | 7265 | 0.0290 | 0.0200 | 0.0380 |
| qwen3_1p7b | gate_capacity_confounded | 43 | alignment_entropy | 0 | 7265 | 0.0290 | 0.0202 | 0.0377 |
| qwen3_1p7b | gate_capacity_confounded | 43 | boundary_mismatch | 0 | 7265 | 0.0290 | 0.0200 | 0.0383 |
| qwen3_1p7b | gate_capacity_confounded | 43 | candidate_count | 1 | 7265 | 0.0290 | 0.0200 | 0.0380 |
| qwen3_1p7b | gate_capacity_confounded | 44 | alignment_bucket | 1-to-1 | 7265 | -0.0194 | -0.0281 | -0.0106 |
| qwen3_1p7b | gate_capacity_confounded | 44 | alignment_entropy | 0 | 7265 | -0.0194 | -0.0281 | -0.0106 |
| qwen3_1p7b | gate_capacity_confounded | 44 | boundary_mismatch | 0 | 7265 | -0.0194 | -0.0285 | -0.0107 |
| qwen3_1p7b | gate_capacity_confounded | 44 | candidate_count | 1 | 7265 | -0.0194 | -0.0279 | -0.0109 |
| qwen3_1p7b | soft_candidates | 42 | alignment_bucket | 1-to-1 | 7161 | 0.0029 | -0.0045 | 0.0106 |
| qwen3_1p7b | soft_candidates | 42 | alignment_bucket | one-to-many | 104 | 0.0673 | 0.0000 | 0.1346 |
| qwen3_1p7b | soft_candidates | 42 | alignment_entropy | (0,0.25] | 104 | 0.0673 | 0.0000 | 0.1346 |
| qwen3_1p7b | soft_candidates | 42 | alignment_entropy | 0 | 7161 | 0.0029 | -0.0045 | 0.0105 |
| qwen3_1p7b | soft_candidates | 42 | boundary_mismatch | 0 | 7265 | 0.0039 | -0.0036 | 0.0113 |
| qwen3_1p7b | soft_candidates | 42 | candidate_count | 1 | 7265 | 0.0039 | -0.0037 | 0.0113 |
| qwen3_1p7b | soft_candidates | 43 | alignment_bucket | 1-to-1 | 7161 | 0.0672 | 0.0570 | 0.0775 |
| qwen3_1p7b | soft_candidates | 43 | alignment_bucket | one-to-many | 104 | 0.0481 | -0.0577 | 0.1538 |
| qwen3_1p7b | soft_candidates | 43 | alignment_entropy | (0,0.25] | 104 | 0.0481 | -0.0577 | 0.1538 |
| qwen3_1p7b | soft_candidates | 43 | alignment_entropy | 0 | 7161 | 0.0672 | 0.0564 | 0.0774 |
| qwen3_1p7b | soft_candidates | 43 | boundary_mismatch | 0 | 7265 | 0.0669 | 0.0564 | 0.0772 |
| qwen3_1p7b | soft_candidates | 43 | candidate_count | 1 | 7265 | 0.0669 | 0.0566 | 0.0771 |
| qwen3_1p7b | soft_candidates | 44 | alignment_bucket | 1-to-1 | 7161 | 0.0228 | 0.0154 | 0.0300 |
| qwen3_1p7b | soft_candidates | 44 | alignment_bucket | one-to-many | 104 | 0.0192 | -0.0288 | 0.0673 |
| qwen3_1p7b | soft_candidates | 44 | alignment_entropy | (0,0.25] | 104 | 0.0192 | -0.0288 | 0.0673 |
| qwen3_1p7b | soft_candidates | 44 | alignment_entropy | 0 | 7161 | 0.0228 | 0.0156 | 0.0300 |
| qwen3_1p7b | soft_candidates | 44 | boundary_mismatch | 0 | 7265 | 0.0227 | 0.0154 | 0.0299 |
| qwen3_1p7b | soft_candidates | 44 | candidate_count | 1 | 7265 | 0.0227 | 0.0156 | 0.0299 |
| tinyllama | entropy_position | 42 | alignment_bucket | one-to-many | 7265 | 0.0228 | 0.0140 | 0.0319 |
| tinyllama | entropy_position | 42 | alignment_entropy | (0,0.25] | 6739 | 0.0217 | 0.0126 | 0.0307 |
| tinyllama | entropy_position | 42 | alignment_entropy | (0.25,0.5] | 526 | 0.0380 | 0.0057 | 0.0722 |
| tinyllama | entropy_position | 42 | boundary_mismatch | (0,1] | 1240 | 0.0323 | 0.0105 | 0.0548 |
| tinyllama | entropy_position | 42 | boundary_mismatch | 0 | 6025 | 0.0209 | 0.0110 | 0.0305 |
| tinyllama | entropy_position | 42 | candidate_count | 1 | 7265 | 0.0228 | 0.0140 | 0.0317 |
| tinyllama | entropy_values | 42 | alignment_bucket | one-to-many | 7265 | 0.0094 | 0.0017 | 0.0172 |
| tinyllama | entropy_values | 42 | alignment_entropy | (0,0.25] | 6739 | 0.0092 | 0.0009 | 0.0177 |
| tinyllama | entropy_values | 42 | alignment_entropy | (0.25,0.5] | 526 | 0.0114 | -0.0190 | 0.0418 |
| tinyllama | entropy_values | 42 | boundary_mismatch | (0,1] | 1240 | 0.0089 | -0.0113 | 0.0282 |
| tinyllama | entropy_values | 42 | boundary_mismatch | 0 | 6025 | 0.0095 | 0.0007 | 0.0181 |
| tinyllama | entropy_values | 42 | candidate_count | 1 | 7265 | 0.0094 | 0.0011 | 0.0175 |
| tinyllama | full_over_gate_only | 42 | alignment_bucket | one-to-many | 7265 | 0.0253 | 0.0162 | 0.0344 |
| tinyllama | full_over_gate_only | 42 | alignment_entropy | (0,0.25] | 6739 | 0.0239 | 0.0145 | 0.0331 |
| tinyllama | full_over_gate_only | 42 | alignment_entropy | (0.25,0.5] | 526 | 0.0437 | 0.0095 | 0.0798 |
| tinyllama | full_over_gate_only | 42 | boundary_mismatch | (0,1] | 1240 | 0.0234 | 0.0024 | 0.0444 |
| tinyllama | full_over_gate_only | 42 | boundary_mismatch | 0 | 6025 | 0.0257 | 0.0159 | 0.0355 |
| tinyllama | full_over_gate_only | 42 | candidate_count | 1 | 7265 | 0.0253 | 0.0162 | 0.0340 |
| tinyllama | full_over_gate_only | 43 | alignment_bucket | one-to-many | 7265 | 0.0451 | 0.0350 | 0.0553 |
| tinyllama | full_over_gate_only | 43 | alignment_entropy | (0,0.25] | 6739 | 0.0459 | 0.0355 | 0.0567 |
| tinyllama | full_over_gate_only | 43 | alignment_entropy | (0.25,0.5] | 526 | 0.0361 | 0.0000 | 0.0741 |
| tinyllama | full_over_gate_only | 43 | boundary_mismatch | (0,1] | 1240 | 0.0540 | 0.0315 | 0.0774 |
| tinyllama | full_over_gate_only | 43 | boundary_mismatch | 0 | 6025 | 0.0433 | 0.0324 | 0.0548 |
| tinyllama | full_over_gate_only | 43 | candidate_count | 1 | 7265 | 0.0451 | 0.0345 | 0.0555 |
| tinyllama | full_over_gate_only | 44 | alignment_bucket | one-to-many | 7265 | 0.0045 | -0.0040 | 0.0136 |
| tinyllama | full_over_gate_only | 44 | alignment_entropy | (0,0.25] | 6739 | 0.0056 | -0.0031 | 0.0148 |
| tinyllama | full_over_gate_only | 44 | alignment_entropy | (0.25,0.5] | 526 | -0.0095 | -0.0418 | 0.0228 |
| tinyllama | full_over_gate_only | 44 | boundary_mismatch | (0,1] | 1240 | 0.0129 | -0.0089 | 0.0347 |
| tinyllama | full_over_gate_only | 44 | boundary_mismatch | 0 | 6025 | 0.0028 | -0.0068 | 0.0126 |
| tinyllama | full_over_gate_only | 44 | candidate_count | 1 | 7265 | 0.0045 | -0.0044 | 0.0134 |
| tinyllama | full_over_static_entropy | 42 | alignment_bucket | one-to-many | 7265 | 0.0154 | 0.0077 | 0.0234 |
| tinyllama | full_over_static_entropy | 42 | alignment_entropy | (0,0.25] | 6739 | 0.0168 | 0.0088 | 0.0251 |
| tinyllama | full_over_static_entropy | 42 | alignment_entropy | (0.25,0.5] | 526 | -0.0019 | -0.0285 | 0.0266 |
| tinyllama | full_over_static_entropy | 42 | boundary_mismatch | (0,1] | 1240 | 0.0097 | -0.0089 | 0.0282 |
| tinyllama | full_over_static_entropy | 42 | boundary_mismatch | 0 | 6025 | 0.0166 | 0.0081 | 0.0249 |
| tinyllama | full_over_static_entropy | 42 | candidate_count | 1 | 7265 | 0.0154 | 0.0076 | 0.0234 |
| tinyllama | full_over_static_entropy | 43 | alignment_bucket | one-to-many | 7265 | 0.0026 | -0.0062 | 0.0116 |
| tinyllama | full_over_static_entropy | 43 | alignment_entropy | (0,0.25] | 6739 | 0.0001 | -0.0089 | 0.0091 |
| tinyllama | full_over_static_entropy | 43 | alignment_entropy | (0.25,0.5] | 526 | 0.0342 | 0.0019 | 0.0665 |
| tinyllama | full_over_static_entropy | 43 | boundary_mismatch | (0,1] | 1240 | 0.0073 | -0.0145 | 0.0290 |
| tinyllama | full_over_static_entropy | 43 | boundary_mismatch | 0 | 6025 | 0.0017 | -0.0076 | 0.0113 |
| tinyllama | full_over_static_entropy | 43 | candidate_count | 1 | 7265 | 0.0026 | -0.0062 | 0.0116 |
| tinyllama | full_over_static_entropy | 44 | alignment_bucket | one-to-many | 7265 | 0.0220 | 0.0127 | 0.0317 |
| tinyllama | full_over_static_entropy | 44 | alignment_entropy | (0,0.25] | 6739 | 0.0242 | 0.0142 | 0.0343 |
| tinyllama | full_over_static_entropy | 44 | alignment_entropy | (0.25,0.5] | 526 | -0.0057 | -0.0399 | 0.0266 |
| tinyllama | full_over_static_entropy | 44 | boundary_mismatch | (0,1] | 1240 | 0.0202 | -0.0040 | 0.0435 |
| tinyllama | full_over_static_entropy | 44 | boundary_mismatch | 0 | 6025 | 0.0224 | 0.0120 | 0.0330 |
| tinyllama | full_over_static_entropy | 44 | candidate_count | 1 | 7265 | 0.0220 | 0.0122 | 0.0317 |
| tinyllama | gate_capacity | 42 | alignment_bucket | 1-to-1 | 7265 | 0.0162 | 0.0063 | 0.0262 |
| tinyllama | gate_capacity | 42 | alignment_entropy | 0 | 7265 | 0.0162 | 0.0062 | 0.0262 |
| tinyllama | gate_capacity | 42 | boundary_mismatch | (0,1] | 1240 | 0.0185 | -0.0040 | 0.0419 |
| tinyllama | gate_capacity | 42 | boundary_mismatch | 0 | 6025 | 0.0158 | 0.0048 | 0.0266 |
| tinyllama | gate_capacity | 42 | candidate_count | 1 | 7265 | 0.0162 | 0.0058 | 0.0263 |
| tinyllama | gate_capacity | 43 | alignment_bucket | 1-to-1 | 7265 | -0.0278 | -0.0377 | -0.0182 |
| tinyllama | gate_capacity | 43 | alignment_entropy | 0 | 7265 | -0.0278 | -0.0376 | -0.0180 |
| tinyllama | gate_capacity | 43 | boundary_mismatch | (0,1] | 1240 | -0.0282 | -0.0516 | -0.0048 |
| tinyllama | gate_capacity | 43 | boundary_mismatch | 0 | 6025 | -0.0277 | -0.0387 | -0.0173 |
| tinyllama | gate_capacity | 43 | candidate_count | 1 | 7265 | -0.0278 | -0.0376 | -0.0182 |
| tinyllama | gate_capacity | 44 | alignment_bucket | 1-to-1 | 7265 | 0.0394 | 0.0296 | 0.0490 |
| tinyllama | gate_capacity | 44 | alignment_entropy | 0 | 7265 | 0.0394 | 0.0296 | 0.0491 |
| tinyllama | gate_capacity | 44 | boundary_mismatch | (0,1] | 1240 | 0.0339 | 0.0105 | 0.0573 |
| tinyllama | gate_capacity | 44 | boundary_mismatch | 0 | 6025 | 0.0405 | 0.0294 | 0.0511 |
| tinyllama | gate_capacity | 44 | candidate_count | 1 | 7265 | 0.0394 | 0.0299 | 0.0493 |
| tinyllama | gate_capacity_confounded | 42 | alignment_bucket | 1-to-1 | 7265 | 0.0022 | -0.0087 | 0.0131 |
| tinyllama | gate_capacity_confounded | 42 | alignment_entropy | 0 | 7265 | 0.0022 | -0.0087 | 0.0135 |
| tinyllama | gate_capacity_confounded | 42 | boundary_mismatch | (0,1] | 1240 | 0.0145 | -0.0129 | 0.0419 |
| tinyllama | gate_capacity_confounded | 42 | boundary_mismatch | 0 | 6025 | -0.0003 | -0.0126 | 0.0118 |
| tinyllama | gate_capacity_confounded | 42 | candidate_count | 1 | 7265 | 0.0022 | -0.0085 | 0.0132 |
| tinyllama | gate_capacity_confounded | 43 | alignment_bucket | 1-to-1 | 7265 | 0.0100 | -0.0011 | 0.0217 |
| tinyllama | gate_capacity_confounded | 43 | alignment_entropy | 0 | 7265 | 0.0100 | -0.0012 | 0.0211 |
| tinyllama | gate_capacity_confounded | 43 | boundary_mismatch | (0,1] | 1240 | 0.0097 | -0.0169 | 0.0363 |
| tinyllama | gate_capacity_confounded | 43 | boundary_mismatch | 0 | 6025 | 0.0101 | -0.0023 | 0.0226 |
| tinyllama | gate_capacity_confounded | 43 | candidate_count | 1 | 7265 | 0.0100 | -0.0012 | 0.0213 |
| tinyllama | gate_capacity_confounded | 44 | alignment_bucket | 1-to-1 | 7265 | 0.0541 | 0.0436 | 0.0646 |
| tinyllama | gate_capacity_confounded | 44 | alignment_entropy | 0 | 7265 | 0.0541 | 0.0432 | 0.0646 |
| tinyllama | gate_capacity_confounded | 44 | boundary_mismatch | (0,1] | 1240 | 0.0492 | 0.0234 | 0.0750 |
| tinyllama | gate_capacity_confounded | 44 | boundary_mismatch | 0 | 6025 | 0.0551 | 0.0435 | 0.0671 |
| tinyllama | gate_capacity_confounded | 44 | candidate_count | 1 | 7265 | 0.0541 | 0.0434 | 0.0646 |
| tinyllama | soft_candidates | 42 | alignment_bucket | one-to-many | 7265 | -0.0494 | -0.0577 | -0.0410 |
| tinyllama | soft_candidates | 42 | alignment_entropy | (0,0.25] | 6739 | -0.0509 | -0.0595 | -0.0424 |
| tinyllama | soft_candidates | 42 | alignment_entropy | (0.25,0.5] | 526 | -0.0304 | -0.0646 | 0.0019 |
| tinyllama | soft_candidates | 42 | boundary_mismatch | (0,1] | 1240 | -0.0484 | -0.0694 | -0.0282 |
| tinyllama | soft_candidates | 42 | boundary_mismatch | 0 | 6025 | -0.0496 | -0.0593 | -0.0403 |
| tinyllama | soft_candidates | 42 | candidate_count | 1 | 7265 | -0.0494 | -0.0581 | -0.0412 |
| tinyllama | soft_candidates | 43 | alignment_bucket | one-to-many | 7265 | 0.0469 | 0.0373 | 0.0567 |
| tinyllama | soft_candidates | 43 | alignment_entropy | (0,0.25] | 6739 | 0.0470 | 0.0369 | 0.0571 |
| tinyllama | soft_candidates | 43 | alignment_entropy | (0.25,0.5] | 526 | 0.0456 | 0.0133 | 0.0798 |
| tinyllama | soft_candidates | 43 | boundary_mismatch | (0,1] | 1240 | 0.0476 | 0.0242 | 0.0702 |
| tinyllama | soft_candidates | 43 | boundary_mismatch | 0 | 6025 | 0.0468 | 0.0359 | 0.0579 |
| tinyllama | soft_candidates | 43 | candidate_count | 1 | 7265 | 0.0469 | 0.0372 | 0.0566 |
| tinyllama | soft_candidates | 44 | alignment_bucket | one-to-many | 7265 | 0.0651 | 0.0535 | 0.0765 |
| tinyllama | soft_candidates | 44 | alignment_entropy | (0,0.25] | 6739 | 0.0656 | 0.0537 | 0.0770 |
| tinyllama | soft_candidates | 44 | alignment_entropy | (0.25,0.5] | 526 | 0.0589 | 0.0171 | 0.1027 |
| tinyllama | soft_candidates | 44 | boundary_mismatch | (0,1] | 1240 | 0.0685 | 0.0403 | 0.0960 |
| tinyllama | soft_candidates | 44 | boundary_mismatch | 0 | 6025 | 0.0644 | 0.0518 | 0.0763 |
| tinyllama | soft_candidates | 44 | candidate_count | 1 | 7265 | 0.0651 | 0.0540 | 0.0764 |
| tinyllama | static_entropy | 42 | alignment_bucket | one-to-many | 7265 | 0.0615 | 0.0505 | 0.0727 |
| tinyllama | static_entropy | 42 | alignment_entropy | (0,0.25] | 6739 | 0.0625 | 0.0509 | 0.0740 |
| tinyllama | static_entropy | 42 | alignment_entropy | (0.25,0.5] | 526 | 0.0494 | 0.0076 | 0.0894 |
| tinyllama | static_entropy | 42 | boundary_mismatch | (0,1] | 1240 | 0.0766 | 0.0492 | 0.1040 |
| tinyllama | static_entropy | 42 | boundary_mismatch | 0 | 6025 | 0.0584 | 0.0461 | 0.0704 |
| tinyllama | static_entropy | 42 | candidate_count | 1 | 7265 | 0.0615 | 0.0505 | 0.0724 |
| tinyllama | static_entropy | 43 | alignment_bucket | one-to-many | 7265 | 0.0056 | -0.0019 | 0.0131 |
| tinyllama | static_entropy | 43 | alignment_entropy | (0,0.25] | 6739 | 0.0064 | -0.0013 | 0.0142 |
| tinyllama | static_entropy | 43 | alignment_entropy | (0.25,0.5] | 526 | -0.0038 | -0.0285 | 0.0209 |
| tinyllama | static_entropy | 43 | boundary_mismatch | (0,1] | 1240 | 0.0089 | -0.0081 | 0.0258 |
| tinyllama | static_entropy | 43 | boundary_mismatch | 0 | 6025 | 0.0050 | -0.0033 | 0.0131 |
| tinyllama | static_entropy | 43 | candidate_count | 1 | 7265 | 0.0056 | -0.0017 | 0.0132 |
| tinyllama | static_entropy | 44 | alignment_bucket | one-to-many | 7265 | -0.0285 | -0.0374 | -0.0194 |
| tinyllama | static_entropy | 44 | alignment_entropy | (0,0.25] | 6739 | -0.0286 | -0.0378 | -0.0190 |
| tinyllama | static_entropy | 44 | alignment_entropy | (0.25,0.5] | 526 | -0.0266 | -0.0589 | 0.0057 |
| tinyllama | static_entropy | 44 | boundary_mismatch | (0,1] | 1240 | -0.0266 | -0.0484 | -0.0056 |
| tinyllama | static_entropy | 44 | boundary_mismatch | 0 | 6025 | -0.0289 | -0.0388 | -0.0188 |
| tinyllama | static_entropy | 44 | candidate_count | 1 | 7265 | -0.0285 | -0.0373 | -0.0194 |

Buckets are assigned once from the candidate/aligner diagnostics and the same sample keys are used for both methods in each paired delta.

## Cross-seed and cross-pair clustered inference

| contrast | pairs | positive pairs | delta | cluster CI low | cluster CI high | aggregate McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| c2c_longest_vs_receiver | 4 | 4 | 0.0929 | 0.0768 | 0.1077 | 0.0000 |
| entropy_position | 1 | 1 | 0.0228 | 0.0139 | 0.0318 | 0.0000 |
| entropy_values | 1 | 1 | 0.0094 | 0.0011 | 0.0175 | 0.0245 |
| full_over_gate_only | 4 | 2 | 0.0119 | -0.0092 | 0.0331 | 0.0000 |
| full_over_hard_span | 4 | 3 | 0.0154 | -0.0114 | 0.0405 | 0.0000 |
| full_over_static_entropy | 1 | 1 | 0.0134 | 0.0028 | 0.0232 | 0.0000 |
| gate_capacity | 1 | 1 | 0.0093 | -0.0266 | 0.0390 | 0.0015 |
| gate_capacity_confounded | 4 | 2 | 0.0035 | -0.0115 | 0.0213 | 0.0111 |
| hard_span_vs_longest | 4 | 4 | 0.0077 | -0.0055 | 0.0198 | 0.0000 |
| hard_span_vs_receiver | 4 | 4 | 0.0923 | 0.0696 | 0.1125 | 0.0000 |
| soft_candidates | 4 | 3 | 0.0129 | -0.0061 | 0.0358 | 0.0000 |
| static_entropy | 1 | 1 | 0.0129 | -0.0278 | 0.0598 | 0.0000 |

## Final B6 gate

| contrast | available pairs | positive pairs | aggregate CI positive | status |
| --- | --- | --- | --- | --- |
| B6_vs_B2 | 4 | 3 | False | fail |
| B6_vs_B5 | 4 | 2 | False | fail |
| combined_B6_vs_B2_and_B5 | 4 | None | False | fail |

## Post-hoc token/head gate diagnostics

| pair | method | seed | status | processed | gate examples | layers | heads rows | token bins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama32_1b | B5 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| llama32_1b | B5 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| llama32_1b | B5 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| llama32_1b | B6 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| llama32_1b | B6 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| llama32_1b | B6 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen25_0p5b | B5 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen25_0p5b | B5 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen25_0p5b | B5 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen25_0p5b | B6 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen25_0p5b | B6 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen25_0p5b | B6 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen3_1p7b | B5 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen3_1p7b | B5 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen3_1p7b | B5 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen3_1p7b | B6 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen3_1p7b | B6 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| qwen3_1p7b | B6 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B5 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B5 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B5 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B6 | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B6 | 43 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B6 | 44 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B6-constant | 42 | ok | 64 | 64 | 28 | 224 | 10 |
| tinyllama | B6-shuffle | 42 | ok | 64 | 64 | 28 | 224 | 10 |

| pair | method | seed | stage | K/V | mean | std | sat low | sat high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama32_1b | B5 | 42 | early | key | 0.8876 | 0.0905 | 0.0000 | 0.2680 |
| llama32_1b | B5 | 42 | early | value | 0.9296 | 0.0029 | 0.0000 | 0.0008 |
| llama32_1b | B5 | 42 | late | key | 0.9320 | 0.0267 | 0.0000 | 0.2459 |
| llama32_1b | B5 | 42 | late | value | 0.9402 | 0.0156 | 0.0000 | 0.2207 |
| llama32_1b | B5 | 42 | middle | key | 0.9076 | 0.0377 | 0.0000 | 0.0362 |
| llama32_1b | B5 | 42 | middle | value | 0.9312 | 0.0092 | 0.0000 | 0.0102 |
| llama32_1b | B5 | 43 | early | key | 0.8701 | 0.1082 | 0.0000 | 0.3169 |
| llama32_1b | B5 | 43 | early | value | 0.9297 | 0.0026 | 0.0000 | 0.0006 |
| llama32_1b | B5 | 43 | late | key | 0.9306 | 0.0232 | 0.0000 | 0.2003 |
| llama32_1b | B5 | 43 | late | value | 0.9403 | 0.0168 | 0.0000 | 0.2327 |
| llama32_1b | B5 | 43 | middle | key | 0.9120 | 0.0351 | 0.0000 | 0.0905 |
| llama32_1b | B5 | 43 | middle | value | 0.9325 | 0.0079 | 0.0000 | 0.0303 |
| llama32_1b | B5 | 44 | early | key | 0.8762 | 0.0970 | 0.0000 | 0.2394 |
| llama32_1b | B5 | 44 | early | value | 0.9300 | 0.0028 | 0.0000 | 0.0008 |
| llama32_1b | B5 | 44 | late | key | 0.9276 | 0.0283 | 0.0000 | 0.2244 |
| llama32_1b | B5 | 44 | late | value | 0.9407 | 0.0161 | 0.0000 | 0.2278 |
| llama32_1b | B5 | 44 | middle | key | 0.9116 | 0.0395 | 0.0000 | 0.1517 |
| llama32_1b | B5 | 44 | middle | value | 0.9314 | 0.0096 | 0.0000 | 0.0148 |
| llama32_1b | B6 | 42 | early | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 42 | early | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 42 | late | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 42 | late | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 42 | middle | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 42 | middle | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 43 | early | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 43 | early | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 43 | late | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 43 | late | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 43 | middle | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 43 | middle | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 44 | early | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 44 | early | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 44 | late | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 44 | late | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 44 | middle | key | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| llama32_1b | B6 | 44 | middle | value | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| qwen25_0p5b | B5 | 42 | early | key | 0.8718 | 0.1228 | 0.0000 | 0.3678 |
| qwen25_0p5b | B5 | 42 | early | value | 0.9296 | 0.0025 | 0.0000 | 0.0002 |
| qwen25_0p5b | B5 | 42 | late | key | 0.9308 | 0.0260 | 0.0000 | 0.2396 |
| qwen25_0p5b | B5 | 42 | late | value | 0.9439 | 0.0171 | 0.0000 | 0.3179 |
| qwen25_0p5b | B5 | 42 | middle | key | 0.8877 | 0.0674 | 0.0000 | 0.0919 |
| qwen25_0p5b | B5 | 42 | middle | value | 0.9310 | 0.0109 | 0.0000 | 0.0214 |
| qwen25_0p5b | B5 | 43 | early | key | 0.8683 | 0.1202 | 0.0000 | 0.3073 |
| qwen25_0p5b | B5 | 43 | early | value | 0.9300 | 0.0022 | 0.0000 | 0.0007 |
| qwen25_0p5b | B5 | 43 | late | key | 0.9333 | 0.0230 | 0.0000 | 0.2228 |
| qwen25_0p5b | B5 | 43 | late | value | 0.9397 | 0.0180 | 0.0000 | 0.2342 |
| qwen25_0p5b | B5 | 43 | middle | key | 0.8976 | 0.0603 | 0.0000 | 0.0686 |
| qwen25_0p5b | B5 | 43 | middle | value | 0.9318 | 0.0084 | 0.0000 | 0.0134 |
| qwen25_0p5b | B5 | 44 | early | key | 0.8492 | 0.1139 | 0.0000 | 0.1873 |
| qwen25_0p5b | B5 | 44 | early | value | 0.9295 | 0.0027 | 0.0000 | 0.0006 |
| qwen25_0p5b | B5 | 44 | late | key | 0.9261 | 0.0280 | 0.0000 | 0.1528 |
| qwen25_0p5b | B5 | 44 | late | value | 0.9425 | 0.0170 | 0.0000 | 0.2881 |
| qwen25_0p5b | B5 | 44 | middle | key | 0.9017 | 0.0640 | 0.0000 | 0.1561 |
| qwen25_0p5b | B5 | 44 | middle | value | 0.9310 | 0.0090 | 0.0000 | 0.0218 |
| qwen25_0p5b | B6 | 42 | early | key | 0.9996 | 0.0171 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 42 | early | value | 0.9996 | 0.0136 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 42 | late | key | 0.9996 | 0.0139 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 42 | late | value | 0.9996 | 0.0132 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 42 | middle | key | 0.9996 | 0.0147 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 42 | middle | value | 0.9996 | 0.0138 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 43 | early | key | 0.9996 | 0.0166 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 43 | early | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 43 | late | key | 0.9996 | 0.0139 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 43 | late | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 43 | middle | key | 0.9996 | 0.0146 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 43 | middle | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 44 | early | key | 0.9996 | 0.0161 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 44 | early | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 44 | late | key | 0.9996 | 0.0139 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 44 | late | value | 0.9996 | 0.0135 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 44 | middle | key | 0.9996 | 0.0150 | 0.0000 | 0.9993 |
| qwen25_0p5b | B6 | 44 | middle | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| qwen3_1p7b | B5 | 42 | early | key | 0.8801 | 0.0975 | 0.0000 | 0.2971 |
| qwen3_1p7b | B5 | 42 | early | value | 0.9295 | 0.0026 | 0.0000 | 0.0000 |
| qwen3_1p7b | B5 | 42 | late | key | 0.9222 | 0.0262 | 0.0000 | 0.1121 |
| qwen3_1p7b | B5 | 42 | late | value | 0.9359 | 0.0316 | 0.0000 | 0.2868 |
| qwen3_1p7b | B5 | 42 | middle | key | 0.9055 | 0.0503 | 0.0000 | 0.0815 |
| qwen3_1p7b | B5 | 42 | middle | value | 0.9298 | 0.0121 | 0.0000 | 0.0335 |
| qwen3_1p7b | B5 | 43 | early | key | 0.8800 | 0.0933 | 0.0000 | 0.2676 |
| qwen3_1p7b | B5 | 43 | early | value | 0.9294 | 0.0028 | 0.0000 | 0.0005 |
| qwen3_1p7b | B5 | 43 | late | key | 0.9266 | 0.0282 | 0.0000 | 0.1583 |
| qwen3_1p7b | B5 | 43 | late | value | 0.9364 | 0.0303 | 0.0000 | 0.3109 |
| qwen3_1p7b | B5 | 43 | middle | key | 0.9091 | 0.0385 | 0.0000 | 0.0797 |
| qwen3_1p7b | B5 | 43 | middle | value | 0.9295 | 0.0118 | 0.0000 | 0.0157 |
| qwen3_1p7b | B5 | 44 | early | key | 0.8564 | 0.0989 | 0.0000 | 0.2221 |
| qwen3_1p7b | B5 | 44 | early | value | 0.9296 | 0.0024 | 0.0000 | 0.0004 |
| qwen3_1p7b | B5 | 44 | late | key | 0.9268 | 0.0256 | 0.0000 | 0.1474 |
| qwen3_1p7b | B5 | 44 | late | value | 0.9327 | 0.0329 | 0.0000 | 0.2743 |
| qwen3_1p7b | B5 | 44 | middle | key | 0.9064 | 0.0404 | 0.0000 | 0.0691 |
| qwen3_1p7b | B5 | 44 | middle | value | 0.9299 | 0.0112 | 0.0000 | 0.0256 |
| qwen3_1p7b | B6 | 42 | early | key | 0.9996 | 0.0167 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 42 | early | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 42 | late | key | 0.9996 | 0.0139 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 42 | late | value | 0.9997 | 0.0135 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 42 | middle | key | 0.9996 | 0.0147 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 42 | middle | value | 0.9996 | 0.0135 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 43 | early | key | 0.9996 | 0.0159 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 43 | early | value | 0.9996 | 0.0136 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 43 | late | key | 0.9996 | 0.0139 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 43 | late | value | 0.9996 | 0.0147 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 43 | middle | key | 0.9996 | 0.0146 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 43 | middle | value | 0.9996 | 0.0136 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 44 | early | key | 0.9996 | 0.0170 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 44 | early | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 44 | late | key | 0.9996 | 0.0141 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 44 | late | value | 0.9996 | 0.0151 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 44 | middle | key | 0.9996 | 0.0147 | 0.0000 | 0.9993 |
| qwen3_1p7b | B6 | 44 | middle | value | 0.9996 | 0.0137 | 0.0000 | 0.9993 |
| tinyllama | B5 | 42 | early | key | 0.8580 | 0.1054 | 0.0000 | 0.1785 |
| tinyllama | B5 | 42 | early | value | 0.9302 | 0.0028 | 0.0000 | 0.0006 |
| tinyllama | B5 | 42 | late | key | 0.9288 | 0.0230 | 0.0000 | 0.1829 |
| tinyllama | B5 | 42 | late | value | 0.9369 | 0.0157 | 0.0000 | 0.1599 |
| tinyllama | B5 | 42 | middle | key | 0.9086 | 0.0483 | 0.0000 | 0.1572 |
| tinyllama | B5 | 42 | middle | value | 0.9315 | 0.0096 | 0.0000 | 0.0105 |
| tinyllama | B5 | 43 | early | key | 0.8911 | 0.0914 | 0.0000 | 0.2732 |
| tinyllama | B5 | 43 | early | value | 0.9295 | 0.0027 | 0.0000 | 0.0005 |
| tinyllama | B5 | 43 | late | key | 0.9269 | 0.0213 | 0.0000 | 0.1499 |
| tinyllama | B5 | 43 | late | value | 0.9390 | 0.0151 | 0.0000 | 0.1799 |
| tinyllama | B5 | 43 | middle | key | 0.9075 | 0.0394 | 0.0000 | 0.0808 |
| tinyllama | B5 | 43 | middle | value | 0.9309 | 0.0083 | 0.0000 | 0.0053 |
| tinyllama | B5 | 44 | early | key | 0.8663 | 0.0997 | 0.0000 | 0.2064 |
| tinyllama | B5 | 44 | early | value | 0.9294 | 0.0026 | 0.0000 | 0.0004 |
| tinyllama | B5 | 44 | late | key | 0.9268 | 0.0267 | 0.0000 | 0.1922 |
| tinyllama | B5 | 44 | late | value | 0.9382 | 0.0194 | 0.0000 | 0.2101 |
| tinyllama | B5 | 44 | middle | key | 0.9088 | 0.0405 | 0.0000 | 0.0788 |
| tinyllama | B5 | 44 | middle | value | 0.9322 | 0.0110 | 0.0000 | 0.0253 |
| tinyllama | B6 | 42 | early | key | 0.9267 | 0.2041 | 0.0000 | 0.8586 |
| tinyllama | B6 | 42 | early | value | 0.9294 | 0.1740 | 0.0000 | 0.8586 |
| tinyllama | B6 | 42 | late | key | 0.9337 | 0.1671 | 0.0000 | 0.8586 |
| tinyllama | B6 | 42 | late | value | 0.9353 | 0.1628 | 0.0000 | 0.8586 |
| tinyllama | B6 | 42 | middle | key | 0.9287 | 0.1830 | 0.0000 | 0.8586 |
| tinyllama | B6 | 42 | middle | value | 0.9304 | 0.1720 | 0.0000 | 0.8586 |
| tinyllama | B6 | 43 | early | key | 0.9243 | 0.2059 | 0.0000 | 0.8586 |
| tinyllama | B6 | 43 | early | value | 0.9292 | 0.1744 | 0.0000 | 0.8586 |
| tinyllama | B6 | 43 | late | key | 0.9312 | 0.1731 | 0.0000 | 0.8586 |
| tinyllama | B6 | 43 | late | value | 0.9371 | 0.1581 | 0.0000 | 0.8586 |
| tinyllama | B6 | 43 | middle | key | 0.9279 | 0.1828 | 0.0000 | 0.8586 |
| tinyllama | B6 | 43 | middle | value | 0.9306 | 0.1713 | 0.0000 | 0.8586 |
| tinyllama | B6 | 44 | early | key | 0.9233 | 0.2056 | 0.0000 | 0.8586 |
| tinyllama | B6 | 44 | early | value | 0.9293 | 0.1743 | 0.0000 | 0.8586 |
| tinyllama | B6 | 44 | late | key | 0.9318 | 0.1729 | 0.0000 | 0.8586 |
| tinyllama | B6 | 44 | late | value | 0.9354 | 0.1626 | 0.0000 | 0.8586 |
| tinyllama | B6 | 44 | middle | key | 0.9288 | 0.1816 | 0.0000 | 0.8586 |
| tinyllama | B6 | 44 | middle | value | 0.9302 | 0.1724 | 0.0000 | 0.8586 |
| tinyllama | B6-constant | 42 | early | key | 0.8530 | 0.1040 | 0.0000 | 0.1667 |
| tinyllama | B6-constant | 42 | early | value | 0.9303 | 0.0030 | 0.0000 | 0.0006 |
| tinyllama | B6-constant | 42 | late | key | 0.9284 | 0.0227 | 0.0000 | 0.1847 |
| tinyllama | B6-constant | 42 | late | value | 0.9375 | 0.0154 | 0.0000 | 0.1669 |
| tinyllama | B6-constant | 42 | middle | key | 0.9114 | 0.0423 | 0.0000 | 0.1254 |
| tinyllama | B6-constant | 42 | middle | value | 0.9318 | 0.0083 | 0.0000 | 0.0114 |
| tinyllama | B6-shuffle | 42 | early | key | 0.9281 | 0.1974 | 0.0000 | 0.8586 |
| tinyllama | B6-shuffle | 42 | early | value | 0.9295 | 0.1737 | 0.0000 | 0.8586 |
| tinyllama | B6-shuffle | 42 | late | key | 0.9367 | 0.1600 | 0.0000 | 0.8586 |
| tinyllama | B6-shuffle | 42 | late | value | 0.9349 | 0.1625 | 0.0000 | 0.8586 |
| tinyllama | B6-shuffle | 42 | middle | key | 0.9303 | 0.1766 | 0.0000 | 0.8586 |
| tinyllama | B6-shuffle | 42 | middle | value | 0.9308 | 0.1708 | 0.0000 | 0.8586 |

Full layer, layer/head, and relative-token-bin K/V statistics are in gate_posthoc_statistics.csv.

## Mechanism conclusions

- [__all__]
- B3-B2=+0.0129, cluster bootstrap CI=[-0.0061, +0.0358]: is inconclusive because the corresponding CI crosses or touches zero.
- B4-B3=+0.0129, cluster bootstrap CI=[-0.0278, +0.0598]: is inconclusive because the corresponding CI crosses or touches zero.
- B5-B2-constant=+0.0093, cluster bootstrap CI=[-0.0266, +0.0390]: is inconclusive because the corresponding CI crosses or touches zero.
- B5-B2=+0.0035, cluster bootstrap CI=[-0.0115, +0.0213]: is inconclusive and additionally confounded by confidence mismatch; B2-constant is required for a clean gate claim.
- B6-constant-confidence=+0.0094, cluster bootstrap CI=[+0.0011, +0.0175]: supports useful entropy information in this counterfactual.
- B6-shuffled-entropy=+0.0228, cluster bootstrap CI=[+0.0139, +0.0318]: supports useful entropy information in this counterfactual.
- Complementarity is inconclusive: B6-B4=+0.0134 and B6-B5=+0.0119, but both corresponding CIs are not strictly above zero.
- [llama32_1b]
- B3-B2=+0.0001 (seed sample std=0.0001), cluster bootstrap CI=[-0.0001, +0.0004]: is inconclusive because the corresponding CI crosses or touches zero.
- B5-B2=+0.0017 (seed sample std=0.0073), cluster bootstrap CI=[-0.0072, +0.0092]: is inconclusive and additionally confounded by confidence mismatch; B2-constant is required for a clean gate claim.
- [qwen25_0p5b]
- B3-B2=-0.0007 (seed sample std=0.0149), cluster bootstrap CI=[-0.0129, +0.0149]: is inconclusive because the corresponding CI crosses or touches zero.
- B5-B2=-0.0039 (seed sample std=0.0285), cluster bootstrap CI=[-0.0245, +0.0279]: is inconclusive and additionally confounded by confidence mismatch; B2-constant is required for a clean gate claim.
- [qwen3_1p7b]
- B3-B2=+0.0312 (seed sample std=0.0324), cluster bootstrap CI=[+0.0047, +0.0659]: supports a contribution from multiple source candidates.
- B5-B2=-0.0059 (seed sample std=0.0305), cluster bootstrap CI=[-0.0280, +0.0280]: is inconclusive and additionally confounded by confidence mismatch; B2-constant is required for a clean gate claim.
- [tinyllama]
- B3-B2=+0.0209 (seed sample std=0.0615), cluster bootstrap CI=[-0.0485, +0.0650]: is inconclusive because the corresponding CI crosses or touches zero.
- B4-B3=+0.0129 (seed sample std=0.0454), cluster bootstrap CI=[-0.0279, +0.0596]: is inconclusive because the corresponding CI crosses or touches zero.
- B5-B2-constant=+0.0093 (seed sample std=0.0341), cluster bootstrap CI=[-0.0260, +0.0381]: is inconclusive because the corresponding CI crosses or touches zero.
- B5-B2=+0.0221 (seed sample std=0.0280), cluster bootstrap CI=[+0.0006, +0.0526]: is positive, but the B2-to-B5 contrast is confounded by confidence mismatch and cannot isolate gate capacity.
- B6-constant-confidence=+0.0094, cluster bootstrap CI=[+0.0015, +0.0173]: supports useful entropy information in this counterfactual.
- B6-shuffled-entropy=+0.0228, cluster bootstrap CI=[+0.0138, +0.0317]: supports useful entropy information in this counterfactual.
- B6 exceeds both B4 (+0.0134) and B5 (+0.0250) with both cluster CIs above zero, supporting complementarity.
- [llama32_1b/B5/seed_42] K/V stage statistics: early/key mean=0.8876, sat-low=0.0000, sat-high=0.2680, early/value mean=0.9296, sat-low=0.0000, sat-high=0.0008, late/key mean=0.9320, sat-low=0.0000, sat-high=0.2459, late/value mean=0.9402, sat-low=0.0000, sat-high=0.2207, middle/key mean=0.9076, sat-low=0.0000, sat-high=0.0362, middle/value mean=0.9312, sat-low=0.0000, sat-high=0.0102; head rows=448, relative-token rows=20.
- [llama32_1b/B5/seed_43] K/V stage statistics: early/key mean=0.8701, sat-low=0.0000, sat-high=0.3169, early/value mean=0.9297, sat-low=0.0000, sat-high=0.0006, late/key mean=0.9306, sat-low=0.0000, sat-high=0.2003, late/value mean=0.9403, sat-low=0.0000, sat-high=0.2327, middle/key mean=0.9120, sat-low=0.0000, sat-high=0.0905, middle/value mean=0.9325, sat-low=0.0000, sat-high=0.0303; head rows=448, relative-token rows=20.
- [llama32_1b/B5/seed_44] K/V stage statistics: early/key mean=0.8762, sat-low=0.0000, sat-high=0.2394, early/value mean=0.9300, sat-low=0.0000, sat-high=0.0008, late/key mean=0.9276, sat-low=0.0000, sat-high=0.2244, late/value mean=0.9407, sat-low=0.0000, sat-high=0.2278, middle/key mean=0.9116, sat-low=0.0000, sat-high=0.1517, middle/value mean=0.9314, sat-low=0.0000, sat-high=0.0148; head rows=448, relative-token rows=20.
- [llama32_1b/B6/seed_42] K/V stage statistics: early/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, early/value mean=1.0000, sat-low=0.0000, sat-high=1.0000, late/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, late/value mean=1.0000, sat-low=0.0000, sat-high=1.0000, middle/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, middle/value mean=1.0000, sat-low=0.0000, sat-high=1.0000; head rows=448, relative-token rows=20.
- [llama32_1b/B6/seed_43] K/V stage statistics: early/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, early/value mean=1.0000, sat-low=0.0000, sat-high=1.0000, late/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, late/value mean=1.0000, sat-low=0.0000, sat-high=1.0000, middle/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, middle/value mean=1.0000, sat-low=0.0000, sat-high=1.0000; head rows=448, relative-token rows=20.
- [llama32_1b/B6/seed_44] K/V stage statistics: early/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, early/value mean=1.0000, sat-low=0.0000, sat-high=1.0000, late/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, late/value mean=1.0000, sat-low=0.0000, sat-high=1.0000, middle/key mean=1.0000, sat-low=0.0000, sat-high=1.0000, middle/value mean=1.0000, sat-low=0.0000, sat-high=1.0000; head rows=448, relative-token rows=20.
- [qwen25_0p5b/B5/seed_42] K/V stage statistics: early/key mean=0.8718, sat-low=0.0000, sat-high=0.3678, early/value mean=0.9296, sat-low=0.0000, sat-high=0.0002, late/key mean=0.9308, sat-low=0.0000, sat-high=0.2396, late/value mean=0.9439, sat-low=0.0000, sat-high=0.3179, middle/key mean=0.8877, sat-low=0.0000, sat-high=0.0919, middle/value mean=0.9310, sat-low=0.0000, sat-high=0.0214; head rows=448, relative-token rows=20.
- [qwen25_0p5b/B5/seed_43] K/V stage statistics: early/key mean=0.8683, sat-low=0.0000, sat-high=0.3073, early/value mean=0.9300, sat-low=0.0000, sat-high=0.0007, late/key mean=0.9333, sat-low=0.0000, sat-high=0.2228, late/value mean=0.9397, sat-low=0.0000, sat-high=0.2342, middle/key mean=0.8976, sat-low=0.0000, sat-high=0.0686, middle/value mean=0.9318, sat-low=0.0000, sat-high=0.0134; head rows=448, relative-token rows=20.
- [qwen25_0p5b/B5/seed_44] K/V stage statistics: early/key mean=0.8492, sat-low=0.0000, sat-high=0.1873, early/value mean=0.9295, sat-low=0.0000, sat-high=0.0006, late/key mean=0.9261, sat-low=0.0000, sat-high=0.1528, late/value mean=0.9425, sat-low=0.0000, sat-high=0.2881, middle/key mean=0.9017, sat-low=0.0000, sat-high=0.1561, middle/value mean=0.9310, sat-low=0.0000, sat-high=0.0218; head rows=448, relative-token rows=20.
- [qwen25_0p5b/B6/seed_42] K/V stage statistics: early/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, early/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/value mean=0.9996, sat-low=0.0000, sat-high=0.9993; head rows=448, relative-token rows=20.
- [qwen25_0p5b/B6/seed_43] K/V stage statistics: early/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, early/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/value mean=0.9996, sat-low=0.0000, sat-high=0.9993; head rows=448, relative-token rows=20.
- [qwen25_0p5b/B6/seed_44] K/V stage statistics: early/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, early/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/value mean=0.9996, sat-low=0.0000, sat-high=0.9993; head rows=448, relative-token rows=20.
- [qwen3_1p7b/B5/seed_42] K/V stage statistics: early/key mean=0.8801, sat-low=0.0000, sat-high=0.2971, early/value mean=0.9295, sat-low=0.0000, sat-high=0.0000, late/key mean=0.9222, sat-low=0.0000, sat-high=0.1121, late/value mean=0.9359, sat-low=0.0000, sat-high=0.2868, middle/key mean=0.9055, sat-low=0.0000, sat-high=0.0815, middle/value mean=0.9298, sat-low=0.0000, sat-high=0.0335; head rows=448, relative-token rows=20.
- [qwen3_1p7b/B5/seed_43] K/V stage statistics: early/key mean=0.8800, sat-low=0.0000, sat-high=0.2676, early/value mean=0.9294, sat-low=0.0000, sat-high=0.0005, late/key mean=0.9266, sat-low=0.0000, sat-high=0.1583, late/value mean=0.9364, sat-low=0.0000, sat-high=0.3109, middle/key mean=0.9091, sat-low=0.0000, sat-high=0.0797, middle/value mean=0.9295, sat-low=0.0000, sat-high=0.0157; head rows=448, relative-token rows=20.
- [qwen3_1p7b/B5/seed_44] K/V stage statistics: early/key mean=0.8564, sat-low=0.0000, sat-high=0.2221, early/value mean=0.9296, sat-low=0.0000, sat-high=0.0004, late/key mean=0.9268, sat-low=0.0000, sat-high=0.1474, late/value mean=0.9327, sat-low=0.0000, sat-high=0.2743, middle/key mean=0.9064, sat-low=0.0000, sat-high=0.0691, middle/value mean=0.9299, sat-low=0.0000, sat-high=0.0256; head rows=448, relative-token rows=20.
- [qwen3_1p7b/B6/seed_42] K/V stage statistics: early/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, early/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/value mean=0.9997, sat-low=0.0000, sat-high=0.9993, middle/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/value mean=0.9996, sat-low=0.0000, sat-high=0.9993; head rows=448, relative-token rows=20.
- [qwen3_1p7b/B6/seed_43] K/V stage statistics: early/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, early/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/value mean=0.9996, sat-low=0.0000, sat-high=0.9993; head rows=448, relative-token rows=20.
- [qwen3_1p7b/B6/seed_44] K/V stage statistics: early/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, early/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, late/value mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/key mean=0.9996, sat-low=0.0000, sat-high=0.9993, middle/value mean=0.9996, sat-low=0.0000, sat-high=0.9993; head rows=448, relative-token rows=20.
- [tinyllama/B5/seed_42] K/V stage statistics: early/key mean=0.8580, sat-low=0.0000, sat-high=0.1785, early/value mean=0.9302, sat-low=0.0000, sat-high=0.0006, late/key mean=0.9288, sat-low=0.0000, sat-high=0.1829, late/value mean=0.9369, sat-low=0.0000, sat-high=0.1599, middle/key mean=0.9086, sat-low=0.0000, sat-high=0.1572, middle/value mean=0.9315, sat-low=0.0000, sat-high=0.0105; head rows=448, relative-token rows=20.
- [tinyllama/B5/seed_43] K/V stage statistics: early/key mean=0.8911, sat-low=0.0000, sat-high=0.2732, early/value mean=0.9295, sat-low=0.0000, sat-high=0.0005, late/key mean=0.9269, sat-low=0.0000, sat-high=0.1499, late/value mean=0.9390, sat-low=0.0000, sat-high=0.1799, middle/key mean=0.9075, sat-low=0.0000, sat-high=0.0808, middle/value mean=0.9309, sat-low=0.0000, sat-high=0.0053; head rows=448, relative-token rows=20.
- [tinyllama/B5/seed_44] K/V stage statistics: early/key mean=0.8663, sat-low=0.0000, sat-high=0.2064, early/value mean=0.9294, sat-low=0.0000, sat-high=0.0004, late/key mean=0.9268, sat-low=0.0000, sat-high=0.1922, late/value mean=0.9382, sat-low=0.0000, sat-high=0.2101, middle/key mean=0.9088, sat-low=0.0000, sat-high=0.0788, middle/value mean=0.9322, sat-low=0.0000, sat-high=0.0253; head rows=448, relative-token rows=20.
- [tinyllama/B6/seed_42] K/V stage statistics: early/key mean=0.9267, sat-low=0.0000, sat-high=0.8586, early/value mean=0.9294, sat-low=0.0000, sat-high=0.8586, late/key mean=0.9337, sat-low=0.0000, sat-high=0.8586, late/value mean=0.9353, sat-low=0.0000, sat-high=0.8586, middle/key mean=0.9287, sat-low=0.0000, sat-high=0.8586, middle/value mean=0.9304, sat-low=0.0000, sat-high=0.8586; head rows=448, relative-token rows=20.
- [tinyllama/B6/seed_43] K/V stage statistics: early/key mean=0.9243, sat-low=0.0000, sat-high=0.8586, early/value mean=0.9292, sat-low=0.0000, sat-high=0.8586, late/key mean=0.9312, sat-low=0.0000, sat-high=0.8586, late/value mean=0.9371, sat-low=0.0000, sat-high=0.8586, middle/key mean=0.9279, sat-low=0.0000, sat-high=0.8586, middle/value mean=0.9306, sat-low=0.0000, sat-high=0.8586; head rows=448, relative-token rows=20.
- [tinyllama/B6/seed_44] K/V stage statistics: early/key mean=0.9233, sat-low=0.0000, sat-high=0.8586, early/value mean=0.9293, sat-low=0.0000, sat-high=0.8586, late/key mean=0.9318, sat-low=0.0000, sat-high=0.8586, late/value mean=0.9354, sat-low=0.0000, sat-high=0.8586, middle/key mean=0.9288, sat-low=0.0000, sat-high=0.8586, middle/value mean=0.9302, sat-low=0.0000, sat-high=0.8586; head rows=448, relative-token rows=20.
- [tinyllama/B6-constant/seed_42] K/V stage statistics: early/key mean=0.8530, sat-low=0.0000, sat-high=0.1667, early/value mean=0.9303, sat-low=0.0000, sat-high=0.0006, late/key mean=0.9284, sat-low=0.0000, sat-high=0.1847, late/value mean=0.9375, sat-low=0.0000, sat-high=0.1669, middle/key mean=0.9114, sat-low=0.0000, sat-high=0.1254, middle/value mean=0.9318, sat-low=0.0000, sat-high=0.0114; head rows=448, relative-token rows=20.
- [tinyllama/B6-shuffle/seed_42] K/V stage statistics: early/key mean=0.9281, sat-low=0.0000, sat-high=0.8586, early/value mean=0.9295, sat-low=0.0000, sat-high=0.8586, late/key mean=0.9367, sat-low=0.0000, sat-high=0.8586, late/value mean=0.9349, sat-low=0.0000, sat-high=0.8586, middle/key mean=0.9303, sat-low=0.0000, sat-high=0.8586, middle/value mean=0.9308, sat-low=0.0000, sat-high=0.8586; head rows=448, relative-token rows=20.

## Diagnostic coverage

| pair | method | seed | task | missing field |
| --- | --- | --- | --- | --- |
| llama32_1b | B0 | 42 | ai2-arc | alignment_bucket |
| llama32_1b | B0 | 42 | mmlu-redux | alignment_bucket |
| llama32_1b | B0 | 42 | openbookqa | alignment_bucket |
| llama32_1b | B0 | 43 | ai2-arc | alignment_bucket |
| llama32_1b | B0 | 43 | mmlu-redux | alignment_bucket |
| llama32_1b | B0 | 43 | openbookqa | alignment_bucket |
| llama32_1b | B0 | 44 | ai2-arc | alignment_bucket |
| llama32_1b | B0 | 44 | mmlu-redux | alignment_bucket |
| llama32_1b | B0 | 44 | openbookqa | alignment_bucket |
| llama32_1b | B1 | 42 | ai2-arc | alignment_bucket |
| llama32_1b | B1 | 42 | mmlu-redux | alignment_bucket |
| llama32_1b | B1 | 42 | openbookqa | alignment_bucket |
| llama32_1b | B1 | 43 | ai2-arc | alignment_bucket |
| llama32_1b | B1 | 43 | mmlu-redux | alignment_bucket |
| llama32_1b | B1 | 43 | openbookqa | alignment_bucket |
| llama32_1b | B1 | 44 | ai2-arc | alignment_bucket |
| llama32_1b | B1 | 44 | mmlu-redux | alignment_bucket |
| llama32_1b | B1 | 44 | openbookqa | alignment_bucket |
| qwen25_0p5b | B0 | 42 | ai2-arc | alignment_bucket |
| qwen25_0p5b | B0 | 42 | mmlu-redux | alignment_bucket |
| qwen25_0p5b | B0 | 42 | openbookqa | alignment_bucket |
| qwen25_0p5b | B0 | 43 | ai2-arc | alignment_bucket |
| qwen25_0p5b | B0 | 43 | mmlu-redux | alignment_bucket |
| qwen25_0p5b | B0 | 43 | openbookqa | alignment_bucket |
| qwen25_0p5b | B0 | 44 | ai2-arc | alignment_bucket |
| qwen25_0p5b | B0 | 44 | mmlu-redux | alignment_bucket |
| qwen25_0p5b | B0 | 44 | openbookqa | alignment_bucket |
| qwen25_0p5b | B1 | 42 | ai2-arc | alignment_bucket |
| qwen25_0p5b | B1 | 42 | mmlu-redux | alignment_bucket |
| qwen25_0p5b | B1 | 42 | openbookqa | alignment_bucket |
| qwen25_0p5b | B1 | 43 | ai2-arc | alignment_bucket |
| qwen25_0p5b | B1 | 43 | mmlu-redux | alignment_bucket |
| qwen25_0p5b | B1 | 43 | openbookqa | alignment_bucket |
| qwen25_0p5b | B1 | 44 | ai2-arc | alignment_bucket |
| qwen25_0p5b | B1 | 44 | mmlu-redux | alignment_bucket |
| qwen25_0p5b | B1 | 44 | openbookqa | alignment_bucket |
| qwen3_1p7b | B0 | 42 | ai2-arc | alignment_bucket |
| qwen3_1p7b | B0 | 42 | mmlu-redux | alignment_bucket |
| qwen3_1p7b | B0 | 42 | openbookqa | alignment_bucket |
| qwen3_1p7b | B0 | 43 | ai2-arc | alignment_bucket |
| qwen3_1p7b | B0 | 43 | mmlu-redux | alignment_bucket |
| qwen3_1p7b | B0 | 43 | openbookqa | alignment_bucket |
| qwen3_1p7b | B0 | 44 | ai2-arc | alignment_bucket |
| qwen3_1p7b | B0 | 44 | mmlu-redux | alignment_bucket |
| qwen3_1p7b | B0 | 44 | openbookqa | alignment_bucket |
| qwen3_1p7b | B1 | 42 | ai2-arc | alignment_bucket |
| qwen3_1p7b | B1 | 42 | mmlu-redux | alignment_bucket |
| qwen3_1p7b | B1 | 42 | openbookqa | alignment_bucket |
| qwen3_1p7b | B1 | 43 | ai2-arc | alignment_bucket |
| qwen3_1p7b | B1 | 43 | mmlu-redux | alignment_bucket |
| qwen3_1p7b | B1 | 43 | openbookqa | alignment_bucket |
| qwen3_1p7b | B1 | 44 | ai2-arc | alignment_bucket |
| qwen3_1p7b | B1 | 44 | mmlu-redux | alignment_bucket |
| qwen3_1p7b | B1 | 44 | openbookqa | alignment_bucket |
| tinyllama | B0 | 42 | ai2-arc | alignment_bucket |
| tinyllama | B0 | 42 | mmlu-redux | alignment_bucket |
| tinyllama | B0 | 42 | openbookqa | alignment_bucket |
| tinyllama | B0 | 43 | ai2-arc | alignment_bucket |
| tinyllama | B0 | 43 | mmlu-redux | alignment_bucket |
| tinyllama | B0 | 43 | openbookqa | alignment_bucket |
| tinyllama | B0 | 44 | ai2-arc | alignment_bucket |
| tinyllama | B0 | 44 | mmlu-redux | alignment_bucket |
| tinyllama | B0 | 44 | openbookqa | alignment_bucket |
| tinyllama | B1 | 42 | ai2-arc | alignment_bucket |
| tinyllama | B1 | 42 | mmlu-redux | alignment_bucket |
| tinyllama | B1 | 42 | openbookqa | alignment_bucket |

## Statistical notes

Paired bootstrap uses 5000 resamples, confidence=0.950, and deterministic base seed 20260717.
Within-pair inference resamples pair/seed clusters then paired examples. Across-pair inference is hierarchical: pairs, then seeds within pair, then paired examples; pairs and seeds are equally weighted.
McNemar p-values use the exact two-sided binomial test over discordant pairs.
Aggregate McNemar pools paired predictions and is not cluster-adjusted; cluster uncertainty is represented by the bootstrap CI.
Receiver reuse is deterministic: same pair+seed, same pair seed 42, same pair nearest seed, any pair same seed, any pair seed 42, then a lexically stable nearest fallback.
Three-seed dispersion is the sample standard deviation (ddof=1); it is reported as missing when only one seed is available.
A lower eval loss is not used in any mechanism conclusion.
