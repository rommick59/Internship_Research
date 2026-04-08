# Report 1 – Literature Review and Dataset Overview - Romain SIAME

## Abstract
This report presents a synthesis of the state of the art on TBM (Tunnel Boring Machine) performance prediction, focusing on the penetration rate (PR), and provides a detailed overview of the available dataset. The objective is to identify the most relevant variables for PR prediction and to lay the groundwork for subsequent modeling work.

## 1. Introduction
Rock excavation is a fundamental step in the construction of many underground infrastructure projects, such as transportation tunnels, sewer networks, or hydraulic galleries. It enables engineers to overcome geological obstacles and create essential spaces for urban and industrial development. In this context, the performance of tunnel boring machines (TBMs) is of paramount importance, as it directly affects productivity, safety, and overall project costs. The key parameters influencing TBM performance include rock strength, rock mass quality, and the operational characteristics of the machine (thrust, torque, rotation speed, face pressure, etc.). Understanding and modeling these parameters is therefore crucial for optimizing excavation operations and anticipating challenges encountered during tunneling.

Predicting the penetration rate (PR) of TBMs is a major challenge in civil engineering, with direct implications for productivity, safety, and project costs. Recent advances in machine learning have enabled more accurate modeling of PR by leveraging both operational and geological data.

## 2. Literature Review
### 2.1. Key Parameters Influencing PR
Recent research highlights several families of parameters that are critical for predicting the penetration rate (PR) of TBMs (Xinbao Yu, 2025; Hongije Yu, 2022; Tao Yan, 2022):

A recent and comprehensive study by Khatti & Mishra (2025) specifically addresses the prediction of penetration rate (PR) for earth pressure balance shield tunnel boring machines (ETBM) in mixed face conditions. The research compares several advanced machine learning and deep learning models—including SVM, SVR, GEP, FFNN, GRU, LSTM, and BiLSTM—using a dataset of 1,197 ETBM events with features such as cutterhead rotation speed (CRS), mean thrust (F/A), mean cutterhead torque (T/D³), upper and lower earth pressure (UEP, LEP), and torque penetration index (TPI). A key novelty is the systematic analysis of multicollinearity among input features using the variance inflation factor (VIF) method, showing that some variables (e.g., LEP and TPI) exhibit considerable multicollinearity, while others (e.g., CRS) show weak multicollinearity. The BiLSTM model achieved the highest predictive accuracy (R = 1.0000 in testing and validation), outperforming other approaches, and was validated through various statistical tests. This research highlights the importance of careful feature selection, multicollinearity analysis, and the use of advanced deep learning models for accurate PR prediction in complex geological conditions.

Key parameters influencing the penetration rate (PR) combine geological descriptors and operational variables: 


**UCS (Uniaxial Compressive Strength):** This is the maximum compressive strength that a rock can withstand before failing. It is a fundamental geotechnical property used to characterize the hardness and drillability of the rock. A higher UCS means the rock is harder and more resistant to penetration, resulting in a lower penetration rate (PR). Conversely, a lower UCS indicates a softer rock, which is easier to excavate and leads to a higher PR. In modern TBM performance modeling, the direct influence of UCS may be reduced in favor of operational parameters, but it remains a key indicator of ground conditions.

**RQD (Rock Quality Designation):** RQD quantifies the degree of fracturing in a rock mass by measuring the percentage of intact core pieces longer than 10 cm in a drill core. A high RQD (close to 100%) indicates a massive, unfractured rock, which is more difficult to excavate and thus lowers the PR. A low RQD (more fractured rock) means the rock mass is broken and easier to penetrate, increasing the PR. RQD is widely used to assess rock mass quality and predict excavation challenges.

**Machine parameters (Thrust, RPM, Torque):**
  - **Thrust:** This is the force applied by the TBM’s hydraulic cylinders to push the cutterhead against the tunnel face. Adequate thrust is essential for effective cutting. If thrust is too low, the cutters may not engage the rock properly, reducing PR. If thrust is too high, it can cause excessive wear or damage to the cutters without improving PR. Optimal thrust maximizes penetration while minimizing wear.
  - **RPM (Cutterhead Rotation Speed, CRS):** RPM controls how fast the cutterhead rotates. Too low an RPM slows down the excavation process, resulting in a low PR. Too high an RPM can reduce the penetration per revolution, also lowering PR, and may cause inefficient cutting or increased wear. The optimal RPM balances speed and cutting efficiency for maximum PR.
  - **Torque:** Torque measures the rotational force required to turn the cutterhead. High torque often indicates that the TBM is encountering hard ground or obstacles, which can lower PR. Moderate, well-adapted torque suggests efficient cutting and favorable ground conditions, supporting a higher PR.

**UEP/LEP (Upper/Lower Earth Pressure):** These are the pressures maintained in the TBM chamber to stabilize the tunnel face and prevent collapse or water ingress. While essential for safety and face stability, their direct impact on PR is limited. However, if the pressures are not properly controlled (too high or too low), they can cause operational problems that indirectly reduce PR.

**Derived indices (TPI, FPI, SE):**
  - **TPI (Torque Penetration Index):** TPI is calculated as the ratio of torque to penetration rate. It measures the amount of torque required to advance the TBM by a unit distance. A high TPI indicates that the machine is facing high resistance (hard ground or inefficient cutting), resulting in a lower PR. A low TPI reflects efficient cutting and favorable conditions, leading to a higher PR.
  - **FPI (Field Penetration Index):** FPI is the ratio of thrust to penetration rate. It quantifies how much thrust is needed to achieve a unit of advance. A high FPI means a lot of force is required for little progress (difficult ground, tool wear, or suboptimal settings), resulting in a low PR. A low FPI indicates efficient excavation and a high PR.
  - **SE (Specific Energy):** SE represents the total energy consumed to excavate a unit volume of ground. It is a comprehensive measure of excavation efficiency. A lower SE means the TBM is excavating efficiently (high PR), while a higher SE indicates more energy is being used for less progress (low PR), often due to hard ground or inefficient machine operation.

### 2.2. Modeling Approaches

- **Empirical/statistical models:** Historically, PR prediction relied on linear models or empirical formulas linking PR to a few geotechnical and machine parameters. These models are simple to implement and interpret but struggle to capture the complexity of nonlinear interactions and multicollinearity between variables.
- **Machine learning and deep learning:** Recent approaches use advanced algorithms (SVM, SVR, LSTM, BiLSTM, GEP, etc.) capable of modeling complex, nonlinear relationships. These models leverage large volumes of operational and geological data and generally outperform classical methods in predictive accuracy, especially in the presence of multicollinearity.
 **Composite indices and robust methods:** Using indices such as TPI, FPI, or SE helps reduce dimensionality and limit the effects of multicollinearity. Robust methods (regularization, deep neural networks) are recommended to improve model stability and generalization.

### 2.3. Key Findings
- CRS (RPM), thrust, torque, UEP/LEP, TPI are most reliable for PR prediction.
- TPI and FPI are strong efficiency indicators.
- Multicollinearity must be managed (Xinbao Yu, 2025).

## 3. Dataset Overview
### 3.1. Data Source
The dataset (TBM data.xlsx) contains operational and derived parameters from TBM operations (source: TBM data.xlsx).

### 3.2. Variables
- **CRS (RPM):** Cutterhead rotation speed (in Revolutions Per Minute)
- **AR (mm/min):** Advance rate (in millimeters per minute)
- **F/A (Mean Thrust):** Mean thrust 
- **T/D³ (Mean Torque):** Normalized torque 
- **UEP/LEP (MPa):** Upper/lower earth pressure (in megapascals)
- **SE (kWh/m³):** Specific energy (in kilowatt-hours per cubic meter)
- **FPI, TPI:** Field and torque penetration indices 
- **PR (mm/r):** Penetration rate (target in millimeters per revolution)

### 3.3. Correlation Analysis
The correlations presented below are calculated from the TBM data.xlsx file.

## 4. Discussion
- The dataset confirms on trends : we have parameters for all the key variables.
- UEP/LEP have limited direct impact on PR but are important for safety, so we have the obligation to keep them.
- High correlation between AR and PR is due to their mathematical relationship.
- Multicollinearity (e.g., F/A and LEP) must be addressed in modeling.

## 5. Project Target
The main objective of this project is to develop the best possible machine learning model for predicting the penetration rate (PR) of TBMs. This involves:
- Leveraging both operational and geological data,
- Applying advanced machine learning algorithms,
- Carefully engineering features and managing multicollinearity,
- Systematically evaluating model performance to achieve the highest predictive accuracy.

The ultimate goal is to provide a robust, data-driven tool to optimize TBM operations.

---

*This report is based on the provided dataset and a synthesis of recent scientific literature. (Xinbao Yu, 2025; Hongije Yu, 2022; Tao Yan, 2022)*

