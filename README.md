# Arrhythmia_Classification_ECG

This is the repository for "Arrhythmia Classification ECG" from HEART DISEASE AI DATATHON 2021 in South Korea. 

- If you have used our code in your research, please cite:
```
https://github.com/DatathonInfo/H.D.A.I.2021
```
# requirements

'''
pip install -r requirements.txt
'''


**Test Method:**
Down below the method is explained in Korean due to the HEART DISEASE AI DATATHON 2021 in South Korea.
'''
python ./3_test.py --project-name small1 --model cnn2d_resnet_v3_small1 --show-roc [roc curve save하고 싶으면 True 아니면 False] --arrhythmia-test-dir [부정맥 폴더 경로] --normal-test-dir [정상 폴더 경로] --cpu [cpu만 쓸거면 1, gpu쓸거면 0]
'''
예시:
'''
python ./3_test.py --project-name small1 --model cnn2d_resnet_v3_small1 --show-roc True --arrhythmia-test-dir ecg/data/validation/arrhythmia --normal-test-dir ecg/data/validation/normal --cpu 0
'''

**Train Method:**
step 1: (preprocess)
'''
python ./1_preprocess.py --data-type [validation 혹은 train] --arrhythmia-data-directory [부정맥 xml 폴더 경로] --normal-data-directory [정상 xml 폴더 경로] --save-directory [정제데이터 저장 경로]
'''

step 2: (train)
'''
python ./2_train.py --project-name [모델 저장 경로] --model cnn2d_resnet_v3_small1
'''
