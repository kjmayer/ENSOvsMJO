

# REPO FOR: Exploring the Relative Contribution of the MJO and ENSO to Midlatitude Subseasonal Predictability with an Interpretable Neural Network

# Transparent Data Use and Repository Guide

Welcome to the repository for exploring the relative contribution of the Madden-Julian Oscillation (MJO) and El Niño Southern Oscillation (ENSO) to midlatitude subseasonal predictability using an interpretable neural network. We prioritize transparent data use, ensuring reproducibility and clarity in our research. The data preparation and preprocessing are detailed in the `./preprocessing` directory, where various notebooks calculate modes and indices, such as SST anomalies, AL index (Z500 anomaly), ENSO index (ONI anomaly), and MJO indices.

## Using This Repository:

Explore our training scripts in the `./training` directory, with the main training script located at `./training/trainANN_gordon.py`. To navigate the key functionality, visit the `./utils` directory. For figures, refer to the corresponding notebooks in the `./interpret` and `./explore` directories.



## File System: 

### To See our training scripts see: 
 - `./training/trainANN_gordon.py` [**main trianing script**]
 - `./training/Run_Bash_Gordon.sh` [**loop through training windows**]
 - `./utils` [**major functionality**]

### Data prep and preprocessing happens here: 
#### [calcultation of modes, and indices]
 - `./preprocessing`
 - - `./preprocessing/Make_SST_Anomaly_CESM2_Vfast.ipynb` (**SST**)
 - - `./preprocessing/Make_Z500_Anomaly_CESM2_Vfast.ipynb` (**AL index**)
 - - `./preprocessing/Make_ONI_Anomaly_CESM2_Vfast.ipynb` (**ENSO index**)
 - - `./preprocessing/Make_MJO_OLR_LIN_METHOD_CESM2_KMWC.ipynb` (**MJO indices**)

## Figure Creation:

- **Figure 1:** `./interpret/interpret-PermutImport.ipynb` and `./interpret/interpret-data-obs.ipynb`
- **Figure 2:** `./interpret/interpret-doy.ipynb` and `./interpret/interpret-doy-plot.ipynb`
- **Figure 3:** `./interpret/interpret-modelcont-plot.ipynb`
- **Figure 4:** `./explore/interpret-data_composite_final.ipynb` (includes Silhouette Plot)

For supplemental figures, check `./explore/ModelBias.ipynb`.

Feel free to explore and contribute to the research. The goal is not just to develop a forecasting system but to utilize an interpretable framework to dissect predictability sources within specific lead times and averaging windows. Initial results indicate a preference for the ENSO state in making accurate predictions at longer subseasonal lead times. Further analysis will uncover the relative contributions of MJO and ENSO to midlatitude subseasonal predictability across various forecast horizons.



### Abstract: 

Forecasting on subseasonal to seasonal (S2S; 2 weeks to 2 months) timescales in the Northern Hemisphere remains a formidable challenge, despite the ongoing development of targeted modeling approaches—both numerical and empirical—over the past decade. The literature has recognized prominent modes of S2S variability, with special emphasis on the Madden-Julian Oscillation (MJO) as a potential stronghold for forecast skill. Recently, there has been a resurgence in literature investigating the subseasonal variability of the El Niño Southern Oscillation (ENSO) teleconnection, highlighting its significant impact in the Northern Hemisphere within the boreal winter season. In this study, our goal is to disentangle midlatitude subseasonal predictive skill that arises from the MJO and ENSO  using an inherently interpretable machine learning model applied to pre-industrial control runs of the Community Earth System Model version 2. This machine learning technique allows us to assess the individual and combined contribution of MJO and ENSO teleconnections to the predictive skill of upper atmospheric circulation over the North Pacific at various forecasting lead times. The aim of this study is not to develop a state-of-the-art forecasting system, but rather to harness a simple, interpretable framework to separate skill from specific sources of predictability within defined forecasting leads and averaging windows. Our initial results show that the machine learning technique generally favors the state of ENSO, rather than the MJO, to make correct predictions on longer subseasonal lead times. Continued analysis will further reveal the relative contributions of these phenomena to midlatitude subseasonal predictability at a range of forecast horizons.
