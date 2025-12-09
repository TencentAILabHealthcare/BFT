## Hunyuan-BFT infer
Create an inference configuration file:
```
model_name_or_path: path/to/bft-sample-1000
template: hunyuan
trust_remote_code: true
```
Run inference API:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 API_PORT=8000 llamafactory-cli api path/to/your_config.yaml
```
Then we can try Hunyuan-BFT:
```
python path/to/api_hunyuan_a13b_health.py
```

## Text response of Hunyuan-BFT

Gene case:
```json
User prompt:

tell me about gene CTSL.

Answer:

**Gene:** **CTSL** (Cathepsin L)  

**Location:** Human chromosome 1p36.33; spans ~13 kb, 13 exons.  

**Protein:** 346‑aa cysteine protease, member of the cathepsin family. Contains an N‑terminal signal peptide, a catalytic domain with the classic cysteine‑histidine‑aspartate triad, and a C‑terminal propeptide that is cleaved for activation.  

**Enzymatic activity:** Hydrolyzes peptide bonds after aromatic or large hydrophobic amino acids (e.g., phenylalanine, tyrosine, tryptophan). Optimal pH ≈ 6.0, but can act at neutral pH in the lysosome.  

**Cellular role:**  
- **Digestive enzyme:** Secreted into the lysosome; degrades extracellular matrix (ECM) proteins, collagen, elastin, and other substrate proteins.  
- **Antigen processing:** Processes peptide fragments for MHC class II presentation.  
- **Apoptosis & signaling:** Cleaves specific substrates that regulate signaling pathways (e.g., BID, p53 regulators).  

**Expression pattern:** High in lysosomes of immune cells (macrophages, dendritic cells, neutrophils) and in epithelial cells of skin, gut, and lung. Also detected in some cancer tissues.  

**Regulation:** Up‑regulated by inflammatory cytokines (IFN‑γ, TNF‑α) and by hypoxia‑inducible factor‑1α (HIF‑1α).  

**Disease associations:**  
- **Cancer:** Over‑expression reported in breast, colorectal, and prostate tumors; promotes invasion, angiogenesis, and immune evasion.  
- **Autoimmunity:** Altered CTSL activity linked to rheumatoid arthritis and osteoarthritis through ECM degradation.  
- **Infectious disease:** Contributes to pathogen‑induced tissue remodeling (e.g., *Mycobacterium tuberculosis*).  

**Genetic variants:** Rare missense mutations (e.g., p.Arg115His) have been identified in patients with congenital lysosomal storage disorders and in some cancer susceptibility studies.  

**Research tools:**  
- **Knock‑out mice:** Viable but show impaired ECM turnover and altered immune responses.  
- **siRNA/CRISPR:** Used to study CTSL’s role in tumor growth and inflammation.  
- **Inhibitors:** Peptidomimetics (e.g., CA‑074511) and small‑molecule inhibitors block CTSL activity in vitro.  

**Key references (selected):**  
1. *J. Biol. Chem.* 2005 – structural analysis of CTSL catalytic domain.  
2. *Nat. Immunol.* 2012 – CTSL in antigen processing.  
3. *Cancer Res.* 2018 – CTSL over‑expression in breast cancer and its impact on metastasis.  

**Bottom line:** CTSL is a lysosomal cysteine protease that digests extracellular matrix and processes antigens, influencing immunity, tissue remodeling, and disease progression, especially cancer.
```

Pathway case:
```json
User prompt:

Your task is to propose a biological process term for gene sets. Here is the gene set: ZMPSTE24, BANF1, WRN, LMNA
    Let do the task step-by-step:
    Step1, write a cirtical analysis for gene functions. For each important point, discribe your reasoning and supporting information.
    Step2, analyze the functional associations among different genes from the critical analysis.
    Step3, summarize a brief name for the most significant biological process of gene set from the functional associations. 
    
    Put the name at the top of analysis as "Process: <name>" and follow the analysis.
    Be concise, do not use unnecessary words.
    Be specific, avoid overly general statements such as "the proteins are involved in various cellular processes".
    Be factual, do not editorialize.
    

Answer:

**Process: Nuclear envelope protein processing and nuclear envelope integrity**

**Step 1 – Critical analysis of gene functions**

- **ZMPSTE24** – Zinc‑metalloprotease that cleaves the C‑terminal of prelamin A, enabling its maturation and incorporation into the nuclear envelope. Essential for nuclear envelope assembly and stability.  
- **BANF1** – Binds lamin A/C and interacts with the nuclear envelope. Promotes nuclear envelope repair and maintains nuclear morphology; loss leads to nuclear envelope defects.  
- **WRN** – Recombinase that resolves DNA replication stress and maintains genome stability. Also participates in nuclear envelope remodeling during S‑phase and in the repair of double‑strand breaks.  
- **LMNA** – Encodes lamin A/C, core structural components of the nuclear lamina. Their proper processing (by ZMPSTE24) and interaction with BANF1 are required for nuclear envelope integrity.

**Step 2 – Functional associations among the genes**

- **ZMPSTE24 ↔ LMNA**: Directly processes LMNA‑derived prelamin A, a prerequisite for lamin A/C incorporation into the nuclear envelope.  
- **LMNA ↔ BANF1**: BANF1 binds lamin A/C, stabilizing the nuclear envelope and facilitating its repair.  
- **WRN ↔ nuclear envelope**: WRN’s role in DNA repair and replication stress resolution involves the nuclear envelope; it also interacts with lamin A/C and contributes to envelope remodeling.  
- **Integrated network**: The four proteins converge on the maintenance of nuclear envelope structure—ZMPSTE24 processes the structural lamin, BANF1 supports its assembly/repair, WRN ensures proper DNA‑related remodeling, and LMNA provides the core architectural component.

**Step 3 – Summary name**

The dominant shared activity is the **processing and preservation of nuclear envelope proteins**, ensuring structural integrity and repair.
```

Health case:
```json
User prompt:


请你通读【病历信息】后，综合该患儿的发病及诊疗过程，回答以下问题：

1、给出该患儿最可能的初步诊断及其诊断依据。

2、进行鉴别诊断。

【病历信息】

性别：男，年龄：15岁，主因"咳嗽2月，呼吸困难15天"于2025-05-01 22:07入病房。

现病史：入院前2月出现咳嗽，病初呈阵发性单声咳嗽，后逐渐加重，夜间为著，近15天时候呼吸困难、夜间睡眠时憋醒，近期上楼费力、爬至4楼时心率增快。病程中无发热，无胸痛，偶有胸闷，夜间偶有喘促，食欲减少，尿便正常。于家中自行口服头孢类药物3-4天，未见好转，为求进一步诊治故来我院，急诊以"重症肺炎、心包积液"收治入院。

既往病史：患儿3年前左脸行血管瘤介入术，现间断介入治疗中。

入院时查体：体温：36.7 ℃，脉搏：98 次/分，呼吸：20 次/分，血压:124/68mmHg，体重：70kg。

神志清楚，精神萎靡，呼吸尚平稳，口唇发绀，周身未见皮疹及出血点咽部充血，右肺呼吸音较弱、肺底可闻及细密湿啰音，心音有力，心率 98次/分，节律规整，腹部平坦、柔软，无固定压痛，无肌紧张及反跳痛，肝脾肋下未触及，肠鸣音4次/分，四肢末梢温暖，活动自如，神经系统查体未见阳性体征。

辅助检查：

2025-05-01肺部CT：右肺感染性病变，右侧心缘旁至右肺下叶背段软组织密度影，心包腔积液。

入我科室时血气分析：PH:7.41 ，PO2：78 mmHg，PaCO2：39 mmHg，BE: 0.1 mmol/L，SO2：96 %，K﹢:3.5 mmol/L，Na﹢: 142 mmol/L，Ca2﹢: 0.81 mmol/L，THbc: 14.0 ，Glu: 5.9 mmol/L，Lac: 0.7 mmol/L。

B型钠尿肽前体（PRO-BNP）测定（2025-05-01 23:24） ：N端前B型钠尿肽测定:60pg/mL

降钙素原检测^血常规+C反应蛋白+血清淀粉样蛋白A（SAA）（2025-05-01 23:52） ：
★白细胞计数:6.00×10⁹/L，中性粒细胞百分比:63.84%，淋巴细胞百分比:25.54%，单核细胞百分比:7.24%，
中性粒细胞绝对值:3.83×10⁹/L，★红细胞计数:4.77×10¹²/L，★血红蛋白:138g/L，★红细胞比容:42.10%，
★血小板计数:258×10⁹/L，C反应蛋白:12.62mg/L↑，血清淀粉样蛋白A:9.43mg/L，降钙素原:0.1ng/mL；

肺炎支原体血清学试验（凝聚法）（2025-05-02 01:07） ：肺炎支原体抗体:阴性。

心电图：窦性心动过速104-110bpm，心电轴正常，不正常心电图，可疑心频图缺血。

心脏三维+左心功能测定+组织超声(江南)(2025-05-02 09:24 ) :卵圆孔未闭心包积液(少-中量)左室张功能欠佳建议随诊复查。

肺炎支原体血清学试验（凝聚法）（2025-05-02 01:07） ：肺炎支原体抗体:阴性；

凝血六项（2025-05-02 09:18） ：*凝血酶原时间:13.50sec，凝血酶原百分率:66.90%，纤维蛋白原时间:4.70sec，*纤维蛋白原含量:4.58g/L，D-二聚体:0.92mg/L FEU。

钙、镁、磷测定^离子三项^心肌酶^肾功测定^肝功1测定（2025-05-02 10:27） ：★天门冬氨酸氨基转移酶:10U/L，★总蛋白:56.52g/L，★白蛋白:37.27g/L，前白蛋白:136.40mg/L，★肌酸激酶:42U/L，缺血性修饰白蛋白:78.30U/mL，★钙:2.00mmol/L；

心肌标志物系列（2025-05-02 10:46） ：肌红蛋白:16.78ng/ml；

癌胚抗原测定(CEA)(化学发光法全自动发光仪)^甲胎蛋白测定(AFP)(化学发光法全自动发光仪)（2025-05-02 10:46） ：正常；

免疫球蛋白测定^抗链球菌O测定（ASO）（免疫法散射比浊法）（2025-05-02 10:50） ：免疫球蛋白L轻链:0.886g/L；

呼吸道病原体核酸检测（八项）（2025-05-02 12:36） ：均阴性。

EB病毒抗体谱（六项）（2025-05-02 13:25） ：EB病毒衣壳抗原IgG:可疑，EB病毒核抗原IgG:阳性；

总IgE测定(酶法)（2025-05-02 14:26） ：总IgE:4.60 TIgEIU/ml；

ABO+RH血型鉴定（2025-05-02 14:30） ：ABO血型（微柱法）:A，RH血型（微柱法）:+；

风湿系列（2025-05-02 14:32） ：类风湿因子:<8.75IU/ml，抗链球菌溶血素"O":<50.3IU/ml；

甲状腺功能测定(五项)（2025-05-02 14:33） ：正常；

血浆凝血因子Ⅱ、Ⅶ、Ⅷ、Ⅸ、Ⅹ活性测定（仪器法）（2025-05-02 14:58） ：凝血因子Ⅶ:52.00%，凝血因子Ⅹ:65.20%；

梅毒螺旋体特异抗体测定(化学发光法全自动发光仪)^乙肝五项^人免疫缺陷病毒抗体测定(Anti-HIV)(化学发光法全自动发光仪)^丙型肝炎抗体测定(Anti-HCV)(化学发光法全自动发光仪)（2025-05-02 15:22） ：★乙肝肝炎病毒表面抗体:169.87mIU/mL；

尿沉渣定量(流式细胞仪法)^尿液分析（2025-05-02 15:35） ：★尿比重:1.010，小圆上皮细胞:3.3个/μL；

1-3-β-D葡聚糖检测+细菌内毒素检测（2025-05-05 13:26） ：1-3-β-D葡聚糖检测:<10.00pg/mL，细菌内毒素测定:<5.00pg/mL；

感染病原宏基因组学检测报告：（痰液）：人疱疹病毒7型（病毒阳性）序列数4，肺炎克雷伯菌（细菌疑似）序列数12，黄曲霉（真菌疑似）序列数1。结核感染T细胞检测(T-SPOT法)：结核感染T细胞检测:阴性。

心脏三维+左心功能测定+组织超声(2025-05-02 09:24 ) :卵圆孔未闭心包积液(少-中量)左室张功能欠佳建议随诊复查

腹部+胃肠道彩色多普勒超声检查(2025-05-02 08:41 ) :目前肝、脾、胰腺结构未见明显异常目前未见急腹症征象

电子支气管镜检查：镜下诊断：主支气管下1/3处至右主支气管可见密集白色结节样突起，右肺上叶、中下叶开口明显狭窄。


Answer:

### 1. 最可能的初步诊断及依据

**初步诊断**：  
- **重症肺炎（右肺感染性病变）**  
- **心包积液（少‑中量）**  
- **卵圆孔未闭**  
- **左室舒张功能欠佳**  

**诊断依据**：  
- **咳嗽2 月，呼吸困难15 天**，夜间加重，活动后气促（上楼费力、爬4楼心率增快）。  
- **肺部CT**显示右肺感染性病变，右侧心缘旁至右肺下叶背段软组织密度影，心包腔积液。  
- **血气分析**：PO₂ 78 mmHg，SO₂ 96 %，提示低氧血症。  
- **BNP 60 pg/mL**（正常＜30 pg/mL），提示容量负荷增加。  
- **心脏超声**：卵圆孔未闭，心包积液（少‑中量），左室张功能欠佳。  
- **心电图**：窦性心动过速（104‑110 bpm），可疑心肌缺血。  
- **炎症指标**：CRP 12.62 mg/L↑，SAA 9.43 mg/L，D‑二聚体 0.92 mg/L，均升高，支持感染和心包填塞风险。  
- **支气管镜**：右主支气管开口狭窄，见白色结节样突起，提示气道阻塞。  

### 2. 鉴别诊断

**（1）支气管哮喘**  
- **支持点**：咳嗽、喘促、夜间加重，BNP升高。  
- **不支持点**：无反复发作史，无明确哮喘家族史，肺功能（未提供）通常表现为可逆性气流受限，而本例CT示右肺实变及心包积液，不符合典型哮喘。  

**（2）肺结核**  
- **支持点**：咳嗽时间长，低热、盗汗（未提及），TB‑SPOT阴性。  
- **不支持点**：TB‑SPOT阴性，CT无结核典型空洞或钙化，痰检未检出结核分枝杆菌。  

**（3）肺癌**  
- **支持点**：CT示右肺占位性病变，年龄15岁（少见）。  
- **不支持点**：肿瘤标志物（CEA、AFP）正常，无体重下降、咯血，病理未证实。  

**（4）心源性哮喘**  
- **支持点**：BNP升高，心包积液，左室舒张功能欠佳，呼吸困难与体位相关。  
- **不支持点**：无心脏病史，心电图无典型心衰表现（如S3、S4），心包积液量少‑中量，未达大量填塞。  

**（5）肺不张**  
- **支持点**：支气管镜示右主支气管开口狭窄，可能致肺不张。  
- **不支持点**：CT示右肺感染性病变，而非单纯不张，且予抗感染治疗后应改善。  

**（6）胸膜炎**  
- **支持点**：胸痛（未提及），D‑二聚体升高。  
- **不支持点**：查体无胸膜摩擦感，CT未示胸膜增厚。  

**（7）病毒性肺炎**  
- **支持点**：呼吸道病原体核酸检测阴性，EBV IgG阳性。  
- **不支持点**：EBV阳性仅提示既往感染，非当前致病。  

**（8）真菌性肺炎**  
- **支持点**：宏基因组学检出黄曲霉。  
- **不支持点**：无免疫抑制史，CT无真菌典型结节，抗真菌治疗（未开始）后应复查。  

**总结**：本例以**重症肺炎合并心包积液**为核心，需首先针对感染治疗，同时关注心包填塞风险及左室功能。鉴别诊断重点在于排除哮喘、结核、肺癌、心源性哮喘、肺不张、胸膜炎、病毒性肺炎及真菌性肺炎。
```