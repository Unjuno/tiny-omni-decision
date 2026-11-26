# Saved text prediction error attribution

All2540 original validation questions retained; selected control step1664.
Read-only diagnostic; no new inference, training, data filtering or checkpoint selection.

| Group | Count | Errors | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| all | 2540 | 977 | 0.615354 | 0.885243 | 0.494998 | 0.067484 |
| type:choice | 1008 | 416 | 0.587302 | 1.042957 | 0.542851 | 0.079121 |
| domain:real estate listings | 18 | 7 | 0.611111 | 0.605820 | 0.355777 | 0.225308 |
| context:complete | 2474 | 950 | 0.616006 | 0.884247 | 0.494502 | 0.069272 |
| teacher:agrees | 2538 | 977 | 0.615051 | 0.885719 | 0.495254 | 0.067712 |
| teacher_gold:at_least_0.8 | 1405 | 400 | 0.715302 | 0.640585 | 0.381647 | 0.036867 |
| type:noul | 987 | 268 | 0.728470 | 0.531945 | 0.358594 | 0.039530 |
| type:score | 545 | 293 | 0.462385 | 1.233368 | 0.653524 | 0.118386 |
| domain:student essay excerpts | 15 | 7 | 0.533333 | 1.107861 | 0.558046 | 0.266982 |
| domain:IoT sensor readings as JSON | 10 | 3 | 0.700000 | 0.950590 | 0.511114 | 0.121880 |
| domain:HR policy questions | 8 | 2 | 0.750000 | 0.592157 | 0.342184 | 0.312584 |
| domain:travel itineraries | 5 | 1 | 0.800000 | 0.537603 | 0.323378 | 0.298119 |
| domain:API error responses | 5 | 3 | 0.400000 | 1.071463 | 0.597820 | 0.437784 |
| teacher_gold:below_0.8 | 1135 | 577 | 0.491630 | 1.188100 | 0.635315 | 0.125926 |
| domain:legal contract clauses | 19 | 6 | 0.684211 | 0.770066 | 0.449320 | 0.190807 |
| domain:restaurant reviews | 11 | 0 | 1.000000 | 0.271858 | 0.127650 | 0.209234 |
| domain:insurance claims | 6 | 2 | 0.666667 | 1.365710 | 0.693927 | 0.499248 |
| domain:translation pairs to be quality-rated | 7 | 4 | 0.428571 | 1.073440 | 0.650114 | 0.411069 |
| domain:e-commerce product reviews | 12 | 3 | 0.750000 | 0.617092 | 0.313526 | 0.118561 |
| domain:hotel booking requests | 14 | 4 | 0.714286 | 0.562314 | 0.339982 | 0.187074 |
| domain:server and application logs | 7 | 3 | 0.571429 | 0.963743 | 0.580468 | 0.248397 |
| domain:academic paper abstracts | 12 | 4 | 0.666667 | 0.577930 | 0.386795 | 0.291164 |
| domain:grant proposals | 13 | 4 | 0.692308 | 0.593216 | 0.346258 | 0.160987 |
| domain:smart home voice commands | 17 | 7 | 0.588235 | 1.151780 | 0.559816 | 0.398923 |
| domain:shipping and logistics updates | 4 | 2 | 0.500000 | 0.958377 | 0.502715 | 0.377469 |
| domain:search queries with a candidate result | 22 | 13 | 0.409091 | 1.154071 | 0.635159 | 0.288391 |
| domain:bug tracker issues | 14 | 5 | 0.642857 | 0.745789 | 0.453156 | 0.303540 |
| domain:app store feedback | 10 | 4 | 0.600000 | 0.844037 | 0.466195 | 0.324192 |
| domain:citations paired with a quoted source | 16 | 5 | 0.687500 | 0.764062 | 0.401506 | 0.257021 |
| domain:product catalogue entries to be matched | 4 | 2 | 0.500000 | 0.850466 | 0.539618 | 0.316550 |
| domain:survey free-text answers | 6 | 2 | 0.666667 | 1.153859 | 0.547101 | 0.393247 |
| domain:game chat messages | 16 | 6 | 0.625000 | 1.045438 | 0.543313 | 0.252561 |
| domain:podcast episode descriptions | 9 | 5 | 0.444444 | 0.939478 | 0.618039 | 0.464070 |
| domain:commit messages with diffs | 19 | 7 | 0.631579 | 0.792914 | 0.453118 | 0.258279 |
| domain:supply chain purchase orders as JSON | 15 | 4 | 0.733333 | 0.534814 | 0.296863 | 0.203817 |
| domain:accessibility audit findings | 39 | 18 | 0.538462 | 1.118747 | 0.630882 | 0.217363 |
| domain:code snippets with a bug report | 25 | 10 | 0.600000 | 0.811180 | 0.503236 | 0.221969 |
| domain:pull request descriptions | 19 | 10 | 0.473684 | 1.109937 | 0.632501 | 0.358852 |
| domain:bank transaction disputes | 4 | 0 | 1.000000 | 0.499458 | 0.267944 | 0.387004 |
| domain:government form submissions | 2 | 0 | 1.000000 | 0.232877 | 0.104268 | 0.187154 |
| domain:chatbot replies to be quality-checked | 4 | 2 | 0.500000 | 0.870723 | 0.457431 | 0.357326 |
| domain:financial news snippets | 4 | 1 | 0.750000 | 0.404418 | 0.265841 | 0.224939 |
| domain:user prompts to be screened for jailbreaks | 5 | 0 | 1.000000 | 0.285767 | 0.150074 | 0.227288 |
| domain:forum posts asking for help | 8 | 2 | 0.750000 | 0.496784 | 0.305119 | 0.124039 |
| domain:medical appointment requests (fictional) | 14 | 3 | 0.785714 | 0.685214 | 0.420374 | 0.363713 |
| domain:SQL queries with a stated intent | 7 | 2 | 0.714286 | 0.739155 | 0.456009 | 0.186397 |
| domain:sales call notes | 3 | 0 | 1.000000 | 0.222556 | 0.111363 | 0.165590 |
| domain:CRM lead records as JSON | 9 | 1 | 0.888889 | 0.362023 | 0.198620 | 0.264386 |
| domain:clinical triage notes (fictional) | 7 | 2 | 0.714286 | 0.652093 | 0.358387 | 0.239361 |
| domain:earnings call excerpts | 12 | 5 | 0.583333 | 1.102600 | 0.610343 | 0.243923 |
| domain:job applications and resumes | 4 | 0 | 1.000000 | 0.360423 | 0.145107 | 0.294703 |
| domain:content moderation queue items | 2 | 0 | 1.000000 | 0.077777 | 0.012086 | 0.074569 |
| domain:social media posts | 10 | 2 | 0.800000 | 0.534755 | 0.289467 | 0.272059 |
| domain:meeting transcripts | 4 | 3 | 0.250000 | 1.311171 | 0.670645 | 0.340599 |
| domain:recipe and nutrition entries | 6 | 2 | 0.666667 | 0.763319 | 0.493139 | 0.433614 |
| domain:RAG passages paired with a user question | 4 | 0 | 1.000000 | 0.408183 | 0.215055 | 0.286567 |
| domain:news headlines with a lede | 8 | 3 | 0.625000 | 0.336548 | 0.242504 | 0.230266 |
| context:truncated | 66 | 27 | 0.590909 | 0.922567 | 0.513601 | 0.110714 |
| domain:sales lead qualification | 35 | 12 | 0.657143 | 0.799282 | 0.429200 | 0.218808 |
| domain:app store review triage for a product team | 25 | 12 | 0.520000 | 0.960979 | 0.549841 | 0.206626 |
| domain:content moderation queue for a video platform | 26 | 12 | 0.538462 | 0.894704 | 0.521112 | 0.245936 |
| domain:HR leave and accommodation requests | 46 | 9 | 0.804348 | 0.602419 | 0.297082 | 0.107376 |
| domain:smart home automation requests | 22 | 10 | 0.545455 | 0.985115 | 0.538477 | 0.240942 |
| domain:medical coding of visit notes (fictional) | 10 | 4 | 0.600000 | 1.252816 | 0.623438 | 0.226588 |
| domain:code review triage | 16 | 8 | 0.500000 | 1.110619 | 0.613450 | 0.348839 |
| domain:hotel guest complaint handling | 28 | 14 | 0.500000 | 0.990633 | 0.591870 | 0.201484 |
| domain:RAG passage grading against a user question | 8 | 4 | 0.500000 | 0.937825 | 0.548113 | 0.284696 |
| domain:payment fraud screening of card transactions | 46 | 24 | 0.478261 | 1.099808 | 0.592006 | 0.198973 |
| domain:search result relevance judgement | 24 | 7 | 0.708333 | 0.865339 | 0.457359 | 0.194388 |
| domain:LLM output guardrail screening | 32 | 15 | 0.531250 | 0.959480 | 0.587583 | 0.171745 |
| domain:translation quality review | 21 | 9 | 0.571429 | 1.065570 | 0.574597 | 0.244572 |
| domain:voice assistant intent resolution | 23 | 7 | 0.695652 | 1.245996 | 0.557475 | 0.326143 |
| domain:manufacturing quality-control reports | 22 | 4 | 0.818182 | 0.497529 | 0.284922 | 0.220345 |
| domain:calendar scheduling conflict resolution | 29 | 14 | 0.517241 | 0.921715 | 0.535020 | 0.184177 |
| domain:industrial IoT sensor anomalies | 40 | 15 | 0.625000 | 0.858597 | 0.464957 | 0.134723 |
| domain:school absence and parent notes | 14 | 8 | 0.428571 | 1.193974 | 0.637504 | 0.226240 |
| domain:open-ended survey response coding | 23 | 13 | 0.434783 | 1.207678 | 0.704423 | 0.270275 |
| domain:multiplayer game chat moderation | 21 | 9 | 0.571429 | 1.135480 | 0.592154 | 0.221882 |
| domain:personal inbox email triage | 9 | 5 | 0.444444 | 1.634524 | 0.788057 | 0.334136 |
| domain:customer record deduplication | 31 | 8 | 0.741935 | 0.708917 | 0.378942 | 0.187022 |
| domain:news desk story routing | 17 | 7 | 0.588235 | 0.658053 | 0.369461 | 0.249819 |
| domain:anti-cheat player reports | 23 | 10 | 0.565217 | 1.048674 | 0.554688 | 0.181853 |
| domain:fleet maintenance scheduling | 24 | 11 | 0.541667 | 1.060822 | 0.604493 | 0.280408 |
| domain:lab sample quality control | 12 | 6 | 0.500000 | 0.964255 | 0.582570 | 0.317051 |
| domain:public benefits eligibility review | 26 | 10 | 0.615385 | 0.901363 | 0.557261 | 0.214958 |
| domain:tutoring session feedback | 22 | 11 | 0.500000 | 0.995985 | 0.585739 | 0.256674 |
| domain:customs declaration review | 22 | 7 | 0.681818 | 0.618650 | 0.391476 | 0.264579 |
| domain:event ticket resale fraud checks | 17 | 4 | 0.764706 | 0.860159 | 0.460947 | 0.361158 |
| domain:building permit review | 19 | 8 | 0.578947 | 1.066230 | 0.549607 | 0.294816 |
| domain:clinical trial eligibility screening (fictional) | 21 | 13 | 0.380952 | 1.456019 | 0.840502 | 0.372034 |
| domain:mortgage document checklist review | 11 | 4 | 0.636364 | 0.814480 | 0.460167 | 0.192177 |
| domain:patent prior-art relevance | 13 | 10 | 0.230769 | 1.563008 | 0.887186 | 0.442839 |
| domain:dataset labelling quality review | 18 | 5 | 0.722222 | 0.687084 | 0.420797 | 0.194787 |
| domain:tax document classification | 19 | 6 | 0.684211 | 0.627624 | 0.408712 | 0.188352 |
| domain:bank wire transfer AML review | 15 | 5 | 0.666667 | 0.844032 | 0.434702 | 0.207265 |
| domain:court filing classification | 38 | 12 | 0.684211 | 0.672213 | 0.392484 | 0.117770 |
| domain:library cataloguing | 18 | 7 | 0.611111 | 0.796673 | 0.490502 | 0.284649 |
| domain:clinical triage intake notes (fictional) | 24 | 9 | 0.625000 | 0.771379 | 0.422112 | 0.148653 |
| domain:genealogy record matching | 14 | 5 | 0.642857 | 1.196990 | 0.599896 | 0.357993 |
| domain:shipment exception handling | 16 | 5 | 0.687500 | 0.611911 | 0.329516 | 0.216234 |
| domain:resume screening for a job opening | 20 | 11 | 0.450000 | 1.095895 | 0.636344 | 0.345025 |
| domain:DevOps change-approval requests | 21 | 9 | 0.571429 | 1.029171 | 0.617415 | 0.247108 |
| domain:ML model monitoring alerts | 33 | 11 | 0.666667 | 0.910949 | 0.476875 | 0.248053 |
| domain:pharmacy refill requests (fictional) | 22 | 10 | 0.545455 | 0.796960 | 0.481615 | 0.196878 |
| domain:site reliability outage postmortems | 20 | 3 | 0.850000 | 0.523569 | 0.284251 | 0.198997 |
| domain:loan underwriting files | 11 | 5 | 0.545455 | 1.171491 | 0.635632 | 0.173000 |
| domain:advertising policy review | 20 | 12 | 0.400000 | 1.191155 | 0.714110 | 0.298455 |
| domain:real estate listing quality checks | 27 | 6 | 0.777778 | 0.691274 | 0.400406 | 0.230641 |
| domain:citation verification against a source excerpt | 14 | 5 | 0.642857 | 0.813217 | 0.440965 | 0.316612 |
| domain:city 311 service requests | 12 | 5 | 0.583333 | 0.653773 | 0.396792 | 0.263104 |
| domain:restaurant health inspection notes | 27 | 10 | 0.629630 | 0.996347 | 0.540952 | 0.236569 |
| domain:podcast and video categorisation | 14 | 5 | 0.642857 | 0.628285 | 0.388473 | 0.305211 |
| domain:meeting notes action-item extraction | 28 | 10 | 0.642857 | 0.756307 | 0.448573 | 0.154145 |
| domain:sports scouting notes | 26 | 13 | 0.500000 | 1.167077 | 0.635575 | 0.210551 |
| domain:freight damage claims | 27 | 14 | 0.481481 | 0.956921 | 0.566953 | 0.223057 |
| domain:fact-check request triage | 17 | 9 | 0.470588 | 1.243741 | 0.676372 | 0.331681 |
| domain:legal intake for a law firm | 14 | 4 | 0.714286 | 0.669491 | 0.353243 | 0.204339 |
| domain:construction site safety reports | 19 | 9 | 0.526316 | 0.735140 | 0.422208 | 0.246115 |
| domain:dating profile moderation | 28 | 15 | 0.464286 | 0.840312 | 0.538305 | 0.233176 |
| domain:museum collection records | 31 | 11 | 0.645161 | 0.726778 | 0.419691 | 0.201902 |
| domain:startup pitch screening for investors | 17 | 10 | 0.411765 | 0.923608 | 0.574376 | 0.229231 |
| domain:KYC identity document review | 15 | 3 | 0.800000 | 0.623090 | 0.352061 | 0.360863 |
| domain:insurance claim triage | 24 | 8 | 0.666667 | 0.919194 | 0.466836 | 0.114207 |
| domain:CI pipeline failure triage | 37 | 15 | 0.594595 | 0.916256 | 0.523357 | 0.203979 |
| domain:data pipeline freshness alerts | 15 | 9 | 0.400000 | 1.354323 | 0.692201 | 0.300776 |
| domain:employee expense report audit | 20 | 9 | 0.550000 | 1.042660 | 0.589675 | 0.294442 |
| domain:research grant proposal review | 23 | 7 | 0.695652 | 0.815154 | 0.452857 | 0.195934 |
| teacher:disagrees | 2 | 0 | 1.000000 | 0.281352 | 0.170592 | 0.221646 |
| domain:contract clause risk review | 18 | 7 | 0.611111 | 1.085645 | 0.571394 | 0.358781 |
| domain:power grid alarm prioritisation | 19 | 8 | 0.578947 | 1.167604 | 0.570786 | 0.234499 |
| domain:tenant maintenance requests | 12 | 1 | 0.916667 | 0.434267 | 0.213014 | 0.282636 |
| domain:prompt-injection screening of retrieved documents | 25 | 9 | 0.640000 | 0.723413 | 0.389653 | 0.202964 |
| domain:farm sensor and crop scouting reports | 34 | 14 | 0.588235 | 0.892771 | 0.514221 | 0.219486 |
| domain:procurement vendor vetting | 10 | 5 | 0.500000 | 1.005552 | 0.585766 | 0.336134 |
| domain:aviation maintenance log entries | 3 | 0 | 1.000000 | 0.492131 | 0.248317 | 0.370066 |
| domain:charity aid request vetting | 9 | 1 | 0.888889 | 0.494286 | 0.287542 | 0.367018 |
| domain:marketplace listing moderation | 15 | 3 | 0.800000 | 0.673913 | 0.361361 | 0.223623 |
| domain:fitness coaching check-ins | 32 | 10 | 0.687500 | 0.963395 | 0.536442 | 0.234808 |
| domain:weather alert prioritisation | 19 | 8 | 0.578947 | 0.914207 | 0.544673 | 0.321921 |
| domain:e-discovery document relevance | 21 | 15 | 0.285714 | 1.487352 | 0.882676 | 0.410033 |
| domain:traffic violation appeal review | 16 | 8 | 0.500000 | 1.064306 | 0.613000 | 0.220507 |
| domain:warehouse inventory discrepancy reports | 33 | 9 | 0.727273 | 0.848717 | 0.430324 | 0.242644 |
| domain:data-quality checks on table rows | 21 | 7 | 0.666667 | 0.886565 | 0.474421 | 0.295508 |
| domain:chatbot reply quality assurance | 24 | 4 | 0.833333 | 0.602640 | 0.328080 | 0.150009 |
| domain:SQL query review against a stated intent | 27 | 8 | 0.703704 | 0.731653 | 0.439723 | 0.208125 |
| domain:product catalogue entity matching | 22 | 11 | 0.500000 | 0.990249 | 0.513603 | 0.221628 |
| domain:product recall notice handling | 18 | 5 | 0.722222 | 0.609456 | 0.372171 | 0.221763 |
| domain:veterinary triage calls | 17 | 7 | 0.588235 | 1.043779 | 0.579713 | 0.347697 |
| domain:academic integrity case review | 26 | 8 | 0.692308 | 0.775649 | 0.450078 | 0.206832 |
| domain:brand-safety review of social posts | 17 | 5 | 0.705882 | 0.694658 | 0.367674 | 0.222469 |
| domain:returns / RMA approval | 13 | 5 | 0.615385 | 1.039896 | 0.584753 | 0.375696 |
| domain:airline disruption rebooking | 10 | 0 | 1.000000 | 0.421169 | 0.210010 | 0.297901 |
| domain:vehicle telematics alerts | 9 | 4 | 0.555556 | 1.129591 | 0.563982 | 0.350691 |
| domain:influencer partnership vetting | 12 | 6 | 0.500000 | 1.098854 | 0.618268 | 0.220751 |
| domain:disaster relief request triage | 4 | 3 | 0.250000 | 1.094114 | 0.696723 | 0.438478 |
| domain:A/B test readout interpretation | 6 | 4 | 0.333333 | 1.018712 | 0.576448 | 0.381261 |
| domain:feature-flag rollout decisions | 12 | 5 | 0.583333 | 1.070996 | 0.584745 | 0.322378 |

## Limits

- Observed checkpoint-selection validation, not blind evaluation
- Teacher disagreement is not proof of label error or a quality ceiling
- Associations across subgroups are not causal effects
- No filtering, reweighting, inference, training or calibration
- Full per-ID attribution and input hashes: `C:/CodexArtifacts/pretrained-reuse-r2/text-error-attribution-01`.
- Teacher distributions are upstream metadata, not a newly measured local model.

## Consequences for the next experiment

Of977 errors,950 occur with complete512-token inputs. Even an oracle fixing
all27 wrong truncated examples, while changing no other predictions, would
raise Accuracy only to62.60%. This is an attribution bound for this saved
prediction set, not an estimate of retraining or long-context model quality.
Increasing the input cap alone is therefore not a sufficient explanation.

Score questions account for293 errors over545 questions (46.24%Accuracy).
191 of these errors are one ordinal position away;102 are two or more away.
Partial credit is not substituted for the original exact-match metric.
Choice questions contribute416 errors, and noul questions268; retain all types.

Upstream teacher top choices agree with gold on2538/2540 questions; the local
model gets both disagreement questions correct. This metadata is not an
independent oracle, and its relation to gold generation is not established here.
It cannot justify calling local errors mislabeled or claiming an upper ceiling.
Teacher gold probability below0.8 correlates with worse local Accuracy, but even
the1405 higher-confidence questions score only71.53%. Do not filter the1135
lower-confidence questions or report the easier subset as general quality.

Do not spend the next run only on longer context or numeric examples. A future
text-learning comparison needs broad original question-format coverage and
explicit held-out-domain diagnostics. No such training was launched by this audit.
The current full objective remains <=200M before quantization and >=95% EACH
for text/image/audio/video; this diagnostic does not meet any product quality gate.

A separate raw-corpus option-key check covered9171 training and1008 validation
dictionary-choice questions: declared criteria keys and upstream probability
keys match in every case. No dropped alternative was found in these corpora.
