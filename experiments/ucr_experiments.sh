#!/bin/bash


vary_ds=('AllGestureWiimoteX' 'AllGestureWiimoteY' 'AllGestureWiimoteZ' 'GestureMidAirD1' 'GestureMidAirD2' 'GestureMidAirD3' 'GesturePebbleZ1' 'GesturePebbleZ2' 'PickupGestureWiimoteZ' 'PLAID' 'ShakeGestureWiimoteZ')
same_length=('Adiac' 'ArrowHead' 'Beef' 'BeetleFly' 'BirdChicken' 'Car' 'CBF' 'ChlorineConcentration' 'CinCECGTorso' 'Coffee' 'Computers' 'CricketX' 'CricketY' 'CricketZ' 'DiatomSizeReduction' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'Earthquakes' 'ECG200' 'ECG5000' 'ECGFiveDays' 'ElectricDevices' 'FaceAll' 'FaceFour' 'FacesUCR' 'FiftyWords' 'Fish' 'FordA' 'FordB' 'GunPoint' 'Ham' 'HandOutlines' 'Haptics' 'Herring' 'InlineSkate' 'InsectWingbeatSound' 'ItalyPowerDemand' 'LargeKitchenAppliances' 'Lightning2' 'Lightning7' 'Mallat' 'Meat' 'MedicalImages' 'MiddlePhalanxOutlineAgeGroup' 'MiddlePhalanxOutlineCorrect' 'MiddlePhalanxTW' 'MoteStrain' 'NonInvasiveFetalECGThorax1' 'NonInvasiveFetalECGThorax2' 'OliveOil' 'OSULeaf' 'PhalangesOutlinesCorrect' 'Phoneme' 'Plane' 'ProximalPhalanxOutlineAgeGroup' 'ProximalPhalanxOutlineCorrect' 'ProximalPhalanxTW' 'RefrigerationDevices' 'ScreenType' 'ShapeletSim' 'ShapesAll' 'SmallKitchenAppliances' 'SonyAIBORobotSurface1' 'SonyAIBORobotSurface2' 'StarLightCurves' 'Strawberry' 'SwedishLeaf' 'Symbols' 'SyntheticControl' 'ToeSegmentation1' 'ToeSegmentation2' 'Trace' 'TwoLeadECG' 'TwoPatterns' 'UWaveGestureLibraryAll' 'UWaveGestureLibraryX' 'UWaveGestureLibraryY' 'UWaveGestureLibraryZ' 'Wafer' 'Wine' 'WordSynonyms' 'Worms' 'WormsTwoClass' 'Yoga' 'ACSF1' 'BME' 'Chinatown' 'Crop' 'DodgerLoopDay' 'DodgerLoopGame' 'DodgerLoopWeekend' 'EOGHorizontalSignal' 'EOGVerticalSignal' 'EthanolLevel' 'FreezerRegularTrain' 'FreezerSmallTrain' 'Fungi' 'GunPointAgeSpan' 'GunPointMaleVersusFemale' 'GunPointOldVersusYoung' 'HouseTwenty' 'InsectEPGRegularTrain' 'InsectEPGSmallTrain' 'MelbournePedestrian' 'MixedShapesRegularTrain' 'MixedShapesSmallTrain' 'PigAirwayPressure' 'PigArtPressure' 'PigCVP' 'PowerCons' 'Rock' 'SemgHandGenderCh2' 'SemgHandMovementCh2' 'SemgHandSubjectCh2' 'SmoothSubspace' 'UMD')

all=('Adiac' 'ArrowHead' 'Beef' 'BeetleFly' 'BirdChicken' 'Car' 'CBF' 'ChlorineConcentration' 'CinCECGTorso' 'Coffee' 'Computers' 'CricketX' 'CricketY' 'CricketZ' 'DiatomSizeReduction' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'Earthquakes' 'ECG200' 'ECG5000' 'ECGFiveDays' 'ElectricDevices' 'FaceAll' 'FaceFour' 'FacesUCR' 'FiftyWords' 'Fish' 'FordA' 'FordB' 'GunPoint' 'Ham' 'HandOutlines' 'Haptics' 'Herring' 'InlineSkate' 'InsectWingbeatSound' 'ItalyPowerDemand' 'LargeKitchenAppliances' 'Lightning2' 'Lightning7' 'Mallat' 'Meat' 'MedicalImages' 'MiddlePhalanxOutlineAgeGroup' 'MiddlePhalanxOutlineCorrect' 'MiddlePhalanxTW' 'MoteStrain' 'NonInvasiveFetalECGThorax1' 'NonInvasiveFetalECGThorax2' 'OliveOil' 'OSULeaf' 'PhalangesOutlinesCorrect' 'Phoneme' 'Plane' 'ProximalPhalanxOutlineAgeGroup' 'ProximalPhalanxOutlineCorrect' 'ProximalPhalanxTW' 'RefrigerationDevices' 'ScreenType' 'ShapeletSim' 'ShapesAll' 'SmallKitchenAppliances' 'SonyAIBORobotSurface1' 'SonyAIBORobotSurface2' 'StarLightCurves' 'Strawberry' 'SwedishLeaf' 'Symbols' 'SyntheticControl' 'ToeSegmentation1' 'ToeSegmentation2' 'Trace' 'TwoLeadECG' 'TwoPatterns' 'UWaveGestureLibraryAll' 'UWaveGestureLibraryX' 'UWaveGestureLibraryY' 'UWaveGestureLibraryZ' 'Wafer' 'Wine' 'WordSynonyms' 'Worms' 'WormsTwoClass' 'Yoga' 'ACSF1' 'AllGestureWiimoteX' 'AllGestureWiimoteY' 'AllGestureWiimoteZ' 'BME' 'Chinatown' 'Crop' 'DodgerLoopDay' 'DodgerLoopGame' 'DodgerLoopWeekend' 'EOGHorizontalSignal' 'EOGVerticalSignal' 'EthanolLevel' 'FreezerRegularTrain' 'FreezerSmallTrain' 'Fungi' 'GestureMidAirD1' 'GestureMidAirD2' 'GestureMidAirD3' 'GesturePebbleZ1' 'GesturePebbleZ2' 'GunPointAgeSpan' 'GunPointMaleVersusFemale' 'GunPointOldVersusYoung' 'HouseTwenty' 'InsectEPGRegularTrain' 'InsectEPGSmallTrain' 'MelbournePedestrian' 'MixedShapesRegularTrain' 'MixedShapesSmallTrain' 'PickupGestureWiimoteZ' 'PigAirwayPressure' 'PigArtPressure' 'PigCVP' 'PLAID' 'PowerCons' 'Rock' 'SemgHandGenderCh2' 'SemgHandMovementCh2' 'SemgHandSubjectCh2' 'ShakeGestureWiimoteZ' 'SmoothSubspace' 'UMD')
n_clusters=(37 3 5 2 2 4 3 3 4 2 2 12 12 12 4 3 2 6 2 2 5 2 7 14 4 14 50 7 2 2 2 2 2 5 2 7 11 2 3 2 7 8 3 10 3 2 6 2 42 42 4 6 2 39 7 3 2 6 3 3 2 60 3 2 2 3 2 15 6 6 2 2 4 2 4 8 8 8 8 2 2 25 5 2 2 10 10 10 10 3 2 24 7 2 2 12 12 4 2 2 18 26 26 26 6 6 2 2 2 2 3 3 10 5 5 10 52 52 52 11 2 4 2 6 5 10 3 3)
len_all=${#all[@]}
len_vary=${#vary[@]}
with_vary=true

if [ "$with_vary" = true ]; then
    for ((i=0;i<len_all;i++)); do
        if [[ ! " ${vary_ds[@]} " =~ " ${all[$i]} " ]]; then
            continue
        fi
        python -m experiments.ucr_pipeline --path ~/datasets/UCRArchive_2018/${all[$i]}/${all[$i]}_TEST.tsv --results-path ./results-paper-algorithms.csv --n-jobs 12 --n-clusters ${n_clusters[$i]} --host odin03 ${@}
    done
else
    for ((i=0;i<len_all;i++)); do
        if [[ " ${vary_ds[@]} " =~ " ${all[$i]} " ]]; then
            continue
        fi
        python -m experiments.ucr_pipeline --path ~/datasets/UCRArchive_2018/${all[$i]}/${all[$i]}_TEST.tsv --results-path ./results-paper-algorithms.csv --n-jobs 12 --n-clusters ${n_clusters[$i]} --equal-lengths --host odin03 ${@}
    done
fi
