
# Exploring the Relative Contribution of the MJO and ENSO to Midlatitude Subseasonal Predictability with an Interpretable Neural Network


## File System: 

### To See our training scripts see: 
 - ./training/trainANN_gordon.py [**main trianing script**]
 - ./training/Run_Bash_Gordon.sh [**loop through training windows**]
 - ./utils [**major functionality**]

### Data prep and preprocessing happens here: 
#### [calcultation of modes, and indices]
 - ./preprocessing
 - - ./preprocessing/Make_SST_Anomaly_CESM2_Vfast.ipynb (**SST**)
 - - ./preprocessing/Make_Z500_Anomaly_CESM2_Vfast.ipynb (**AL index**)
 - -./preprocessing/Make_ONI_Anomaly_CESM2_Vfast.ipynb (**ENSO index**)
 - -./preprocessing/Make_MJO_OLR_LIN_METHOD_CESM2_KMWC.ipynb (**MJO indices**)

### Figure 1
- ./interpret/
### Figure 2
- ./interpret/
### Figure 3
- ./interpret/
### Figure 4
- ./explore/interpret-data_composite_final.ipynb
### Supplemental Figures: 
- ./explore/

### Abstract: 

Forecasting on subseasonal to seasonal (S2S; 2 weeks to 2 months) timescales in the Northern Hemisphere remains a formidable challenge, despite the ongoing development of targeted modeling approaches—both numerical and empirical—over the past decade. The literature has recognized prominent modes of S2S variability, with special emphasis on the Madden-Julian Oscillation (MJO) as a potential stronghold for forecast skill. Recently, there has been a resurgence in literature investigating the subseasonal variability of the El Niño Southern Oscillation (ENSO) teleconnection, highlighting its significant impact in the Northern Hemisphere within the boreal winter season. In this study, our goal is to disentangle midlatitude subseasonal predictive skill that arises from the MJO and ENSO  using an inherently interpretable machine learning model applied to pre-industrial control runs of the Community Earth System Model version 2. This machine learning technique allows us to assess the individual and combined contribution of MJO and ENSO teleconnections to the predictive skill of upper atmospheric circulation over the North Pacific at various forecasting lead times. The aim of this study is not to develop a state-of-the-art forecasting system, but rather to harness a simple, interpretable framework to separate skill from specific sources of predictability within defined forecasting leads and averaging windows. Our initial results show that the machine learning technique generally favors the state of ENSO, rather than the MJO, to make correct predictions on longer subseasonal lead times. Continued analysis will further reveal the relative contributions of these phenomena to midlatitude subseasonal predictability at a range of forecast horizons.
