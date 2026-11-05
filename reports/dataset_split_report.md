# PhytoTrace AI — Dataset Splitting & Leakage Audit Report

**Date:** 2026-11-03 14:45:19  
**Total Dataset Records:** 95 images  
**Leakage Status:** ✅ PASSED (Zero Image, Plant, Leaf, or Duplicate Leakage)

---

## 1. Source & Image Skew Audit

| Source / Farm | Total Images | Dataset Share | Healthy | Leaf Miner | Other Damage | Unique Plants |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ARCHIVE_REF` | 8 | 8.4% | 0 | 5 | 3 | 8 |
| `BLOCK_3` | 81 | 85.3% | 35 | 43 | 3 | 81 |
| `FARM_CANON_01` | 6 | 6.3% | 2 | 0 | 4 | 6 |

> [!NOTE]
> `BLOCK_3` comprises **85.3%** of all available data. A coarse group split on `farm_id` forces 81 images into a single partition and produces an invalid 6/8/81 partition. 
> To preserve statistical validity, splitting is conducted at the **physical plant unit** (`farm_id + plant_id`) with **stratified class balancing**.

---

## 2. Partition Breakdown & Class Representation

| Partition | Total Images | Percentage | Healthy | Leaf Miner Damage | Other Damage | Independent Plants |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **65** | **68.4%** | 25 | 34 | 6 | 65 |
| **Val** | **15** | **15.8%** | 6 | 7 | 2 | 15 |
| **Test** | **15** | **15.8%** | 6 | 7 | 2 | 15 |

---

## 3. Data Leakage Verification Summary

1. **Physical Plant Independence:** Zero plant overlap across partitions (`65 + 15 + 15 = 95` unique plants).
2. **Physical Leaf Independence:** Zero leaf overlap across partitions. Relative local leaf indices (`L01`) on different plants do not cause false-positive leakage.
3. **Duplicate Group Disjointness:** No exact SHA-256 or perceptual near-duplicates span partitions.
4. **Minority Class Representation:** The minority class `other_damage` (10 total samples) is represented in Train (6), Val (2), and Test (2).
