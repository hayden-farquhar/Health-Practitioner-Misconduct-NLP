# Ahpra registrant counts, 2010/11 to 2025/26

Files in this folder:

- `ahpra_registrants_2010_2025.csv`: national registrant counts by profession and report year, plus a `profession=ALL` row per year (304 rows).
- `ahpra_registrants_by_state_2010_2025.csv`: all-profession registrant counts by state or territory of principal place of practice (PPP), plus `No PPP` and `AUSTRALIA_TOTAL`, per year (160 rows).

The file names say 2010_2025, but the data run to 2025/26. Ahpra's 2025/26 annual report (as at 30 June 2026) was already published on the access date, so it is included.

Access date: 2026-11-06. All sources are the Ahpra and National Boards annual reports, downloaded from https://www.ahpra.gov.au/Publications/Annual-reports (current report) and https://www.ahpra.gov.au/Publications/Annual-reports/Annual-report-archive (2009/10 to 2024/25). The `source_url` column gives the exact PDF for each year. No National Board summaries, quarterly registrant data or other secondary sources were used. Every value comes from the main annual report PDF.

## Source table per year

| Report year | As at | Table | PDF page |
|---|---|---|---|
| 2010/11 | 30 Jun 2011 | Table 4 | 48 |
| 2011/12 | 30 Jun 2012 | Table 4 | 74 |
| 2012/13 | 30 Jun 2013 | Table 112 | 128 |
| 2013/14 | 30 Jun 2014 | Table R1 | 112 |
| 2014/15 | 30 Jun 2015 | Table R1 | 31 |
| 2015/16 | 30 Jun 2016 | Table R1 | 40 |
| 2016/17 | 30 Jun 2017 | Table 3 | 38 |
| 2017/18 | 30 Jun 2018 | Table 3 | 38 |
| 2018/19 | 30 Jun 2019 | Table 5 | 55 |
| 2019/20 | 30 Jun 2020 | Table 4 (incl. sub-register); Table 4A (excl.) | 59 |
| 2020/21 | 30 Jun 2021 | Table 4 (incl.); Table 4A (excl.) | 53 |
| 2021/22 | 30 Jun 2022 | Table 8 (incl.); Table 8A (excl.) | 54 |
| 2022/23 | 30 Jun 2023 | Table 7 | 53 |
| 2023/24 | 30 Jun 2024 | Table 7 | 52 |
| 2024/25 | 30 Jun 2025 | Table 7 | 53 |
| 2025/26 | 30 Jun 2026 | Table 6 | 52 |

Page numbers are PDF page indices (1-based), not the printed folio. In every year the table is the "registered practitioners by profession and principal place of practice" table, so the profession counts, state counts and grand total come from the same table.

## Definitions

- **Registrant.** A person holding current registration in a profession at 30 June. This covers every registration type: general, specialist (alone or with general), provisional, limited (including the 2010/11 transitional "limited (public interest – occasional practice)") and non-practising.
- **Non-practising registrants are included in every year.** In 2010/11, the subtotals by registration type in Table 6, which include the non-practising rows, sum exactly to the Table 4 totals. In 2013/14, the registration-type and age tables in Appendix 8 give the same 619,509 total. From 2016/17, the reports say directly that only part of the registrant base holds practising registration: 97.8% (2016/17), 97.9% (2018/19), 98.1% (2019/20), 97.6% (2020/21) and 96.9% (2023/24 and 2024/25). Non-practising registrants are therefore about 2–3% of the totals. The PPP tables do not split out practising-only counts, so a practising-only denominator cannot be built from these tables.
- **Students are excluded.** Student registration began on 31 March 2011 and is reported in separate tables.
- **Principal place of practice (PPP).** This is the basis of the state file. "No PPP" covers practitioners with an overseas or unknown address. The national total includes No PPP.
- **Medical practitioner.** All medical registrants: general, specialist, general plus specialist, provisional (interns), limited and non-practising.
- **Dentist.** The `dentist` key holds the "Dental practitioner" row. This covers all divisions of the Dental Board register: dentists (including dental specialists), dental hygienists, dental prosthetists, dental therapists and oral health therapists. It is not a count of dentists alone. The PPP tables never split dentists out.
- **Podiatrist.** Includes podiatric surgeons (footnoted from 2016/17 onward).
- **Nurses and midwives.** Every report gives three mutually exclusive rows: "Nurse" (nurse only), "Midwife" (midwife only) and "Nurse and midwife" (dual registrants, counted once). These are kept as `nurse_only`, `midwife_only` and `nurse_and_midwife_dual`. The `nurse` key is DERIVED: it is the sum of those three rows, i.e. everyone registered with the Nursing and Midwifery Board, and it is marked "(derived)" in `source_location`. It is not a printed figure. Before 1 December 2018, nursing and midwifery were legally one profession with separate divisions. From 1 December 2018, the National Law recognises them as two professions (2018/19 report, p 55). The table layout and the three rows did not change.
- **ALL row.** The reported grand total. It equals the sum of the profession rows, which was checked for every year. A person registered in two professions (for example nurse and paramedic) is counted once in each, so ALL counts registrations by profession and slightly overcounts unique people. The 2019/20 report notes that "some health practitioners are registered in more than one health profession". Dual nurse-and-midwife registrants are counted once.

## Definitional breaks and caveats for a time series

1. **Profession entry dates.**
   - The original 10 professions were regulated from 1 July 2010.
   - WA joined the National Scheme on 18 October 2010. It is fully counted at 30 June 2011.
   - Aboriginal and Torres Strait Islander health practitioners, Chinese medicine practitioners, medical radiation practitioners and occupational therapists joined on 1 July 2012. They are first counted in 2012/13, and their 2010/11 and 2011/12 rows are blank.
   - Paramedics joined on 1 December 2018. They are first counted in 2018/19 (17,323), and their 2010/11 to 2017/18 rows are blank. 2018/19 is a partial first year for paramedics.
   - These entries cause step changes in the ALL total in 2012/13 (+4 professions, about 33,400 registrants) and 2018/19 (+17,323 paramedics). The 2018/19 report says growth excluding paramedics was 3.5%. For an all-profession rate, use a constant profession set or adjust for these steps.
2. **COVID-19 pandemic response sub-register, 2019/20 to 2021/22.** The sub-register opened on 6 April 2020. In these three years the headline tables include sub-register practitioners, mostly returning medical practitioners, nurses, midwives, pharmacists, physiotherapists, psychologists and diagnostic radiographers. The `registrants` column holds the headline (including) value, as Ahpra reports it. The value excluding the sub-register (Tables 4A/4A/8A) is in `notes` as `excl_pandemic_subregister=N`. The ALL totals were:

   | Year | Including sub-register | Excluding sub-register |
   |---|---|---|
   | 2019/20 | 801,659 | 769,430 |
   | 2020/21 | 825,720 | 801,750 |
   | 2021/22 | 852,272 | 833,318 |

   For psychologists, physiotherapists and medical radiation practitioners, the sub-register closed on 19 April 2021. For the other professions it was extended to April 2022. It formally closed on 8 June 2023, and anyone still on it moved to the main register, so 2022/23 onward has no separate split. The 2021/22 table excluding the sub-register also shows lower dental, OT, optometry and podiatry counts than the including table. These are recorded exactly as printed.
3. **2010/11 state counts were restated.** The 2011/12 report (Table 4, footnote 4) restated every 2010/11 state figure after data cleansing cut "No PPP" from 16,836 to 13,709. For example, NSW went from 156,104 to 156,139. The national total of 530,115 did not change. The state file keeps the figures as originally published and gives the restated value in `notes`. Use the restated values for a state time series.
4. **Small restatement for 2010/11 midwife and dual rows.** The prior-year column of the 2011/12 report shows 1,788 midwives and 40,325 dual registrants. The original 2010/11 report and the 2012/13 report both show 1,789 and 40,324. The original values are kept and the discrepancy is noted on those rows.
5. **The "No PPP" group grows a lot after 2021/22.** It rose from 21,464 in 2021/22 to 47,134 in 2025/26, mostly nurses (No PPP nurses: 13,133 in 2021/22, 34,561 in 2025/26). This reflects overseas-based or unknown-address registrants, likely linked to overseas-qualified nurse registration pathways, though that is an inference and the reports do not state it. These people are in the national totals but in no state. A national rate per 10,000 therefore includes them in the denominator even though few are likely to appear before Australian tribunals.
6. **Point-in-time stock versus decision year.** Counts are a stock at 30 June, the end of the financial year. To convert calendar-year tribunal decision counts to rates, align the years explicitly. For example, use the 30 June count of the same year, or the mean of two adjacent 30 June counts.
7. **Errata checked.** The published correction notices for 2012/13 (8 January 2014) and 2013/14 (7 January 2015) were read. Neither changes any registrant count: they cover board structures, notification counts and board-member lists. The 2016/17 supplementary data tables were re-issued with "remediated" data, but only for endorsements (Table R10). The registrant totals in the main report are unaffected.

## Gaps

- No value is missing within scope. The 16 blank rows are all structural: the four professions that joined in 2012 are blank for 2010/11 and 2011/12 (8 rows), and paramedics are blank for 2010/11 to 2017/18 (8 rows). These professions had no registrants under the National Scheme in those years. The cells are left blank, not zero, and the joining date is given in `notes`.
- A dentist-only count (excluding hygienists, therapists and prosthetists) is not in the PPP tables. Divisional breakdowns exist elsewhere in some reports (for example 2012/13 Table A3, and the 2015/16 supplementary Table R6) but were not extracted.
- Practising-only counts by profession are not available from these tables (see Definitions).
- ACT cells for Aboriginal and Torres Strait Islander health practitioners are blank in the 2022/23, 2023/24 and 2024/25 tables. This does not affect the national or state totals used here.

## Verification performed

- Every value was transcribed from the cited table. A script then checked that each printed value appears on the cited PDF page. It also checked that profession rows sum to the reported grand total for every year (and for the excluding-sub-register tables), and that state and No PPP rows sum to the national total. There were no failures.
- Cross-year check: each report's prior-year column was compared with the previous report's values. The only differences are the 2010/11 restatements described in breaks 3 and 4.
- Spot check: 3 random non-derived rows per report year (48 rows in total) were re-extracted directly from the source PDF page and matched to the row label. All 48 matched, with no mismatches.
