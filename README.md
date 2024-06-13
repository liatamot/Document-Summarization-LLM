# Document-Summarization-LLM

## Team

|![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/e7394268-0f94-4468-8cf5-3cf67e4edd07)|![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9233ab6e-25d5-4c16-8dd4-97a7b8535baf) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9c75cbd9-f409-4fdd-a5c3-dec082ade3bf) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/388eac05-7cd9-4688-8a87-5b6b742715cf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/48dd674c-ab93-48d1-9e05-e7e8e402597c) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/0a524747-a854-4eee-95b6-108c84514df8) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            김영천             |            최장원            |            배창현           |            박성우             |            조예람             |            이소영B             |
|                            팀장                            |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |

## 0. Overview
### Environment
- Vscode, ssh server(RTX 3090/Ubuntu 20.04.6), pytorch

## 1. Competiton Info

### Overview
![image](https://github.com/dudcjs2779/kr-document-type-classification-upstage-competition/assets/42354230/b34c5dea-1683-4d78-84ac-c2e70bc1255b)
해당 대회는 **Upstage AI Lab** 과정에서 비공개로 진행된 내부 대회이며 **일상 대화에 대한 요약**을 효과적으로 생성하는 모델을 개발하는 대회입니다. 해당 대회에서 주어진 데이터셋은 영어 일상 대화 요약 Task에서 많이 활용되는 [**Dialogsum**](https://huggingface.co/datasets/knkarthick/dialogsum) 데이터셋을 **한국어로 번역한** 데이터라는 점이 대회의 특징입니다.

### Evaluation
요약문을 정확하게 평가할 수 있는 평가지표를 설계하는 것은 매우 어렵습니다. 왜나하면 요약문은 **관점에 따라서 다르게 요약**이 될 수 있기 때문입니다. 따라서 해당 대회에서는 예측된 요약 문장을 **3개의 정답 요약 문장과 비교하여** metric의 평균 점수를 산출합니다. 

본 대회에서는 ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1, 총 3가지 종류의 metric으로부터 산출된 평균 점수를 더하여 최종 점수를 계산합니다. DialogSum 데이터셋은 Multi-Reference Dataset으로 multi-reference에 대한 average를 보는 것이 중요합니다. 따라서 데이터셋의 특성에 맞추어 **최종 점수 산출도 평균을 활용**했습니다. ROUGE 스코어는 단순히 예측된 요약문과 정답 요약문을 비교하여 맞춘 단어 갯수를 비교하는 평가지표입니다. 최종스코어 산출 방식은 아래와 같습니다.
![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/807b16a7-6733-4e4d-9623-2aebbd4eeecf)


### Timeline
- 2024.03.08 ~ 2024.03.20
  

## 2. Data descrption

### Dataset overview
![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/a072eae5-c709-4a2a-81d4-95784c230234)
해당 데이터는 영어 일상 대화 데이터셋은 [**Dialogsum**](https://huggingface.co/datasets/knkarthick/dialogsum) 데이터셋을 **한국어로 번역한** 데이터입니다. **대화문과 요약문**을 포함하고 있으며 이러한 **비정형 텍스트 데이터**를 고려하여 모델을 훈련하고, 요약문의 생성 성능을 높이기 위한 최적의 방법을 찾아야 합니다. 

Train Data 12457개, Valid Data 499개, Test Data 499개로 이루어져 있으며 발화자 및 여러 개인정보에 대해서 **마스킹 처리**가 돼 있는 것 또한 해당 데이터셋의 특징입니다.


### EDA
#### 샘플데이터
<img src="https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/c6d69ed8-0845-4d06-9678-6e1add5f3768" width="512">

#### Train, Valid, Test 대화문 및 요약문 길이 비교
<img src="https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/f933c578-eb74-437d-b6cd-acf720f1dc4b" width="330">
<img src="https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/97322551-cc5f-4081-a628-b0ca57b66429" width="330">
<img src="https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/adb53049-bbc3-4c87-9b98-b93d3814a7b2" width="330"> 

#### Data cleansing
![05](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/d05f07e5-b7da-485b-9a43-0d099aa11abd)
정규식을 활용하여 잘못 마스킹된 데이터 및 순서가 이상한 데이터를 수정하였습니다.

### Data Processing
<img src="https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/6f521f47-8e7c-4f80-8c63-acc5345f79df" width="1024">  

**GPT3.5 Api를 사용하여 데이터를 합성했습니다.** 위 그림과 같이 GPT에게 Instruction을 주고 학습데이터에서 샘플링한 5개의 대화문-요약문 쌍데이터를 예시로 보여준 뒤 질문으로 1개의 대화문을 입력해 요약문을 생성했습니다.  

SAMSUM 데이터셋 활용
![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/514435f1-c572-4db3-af00-364e1a2013cd)

대회 데이터셋이 번역 데이터라는 점을 활용하여 **다른 영어 대화문 데이터셋을 한글로 번역**하여 학습 데이터셋으로 활용하였습니다. 해당 데이터셋이 Upstage에서 추최하는 대회이기에 **Solar Api**를 활용하여 번역을 진행하면 좋을 것이라고 판단해서 Solar Api를 활용해 번역을 진행하였습니다.

## 3. Modeling

### Model descrition
[digit82/kobart-summarization](https://huggingface.co/digit82/kobart-summarization)
- 기본적으로 좋은 성능을 보여줌

[eenzeenee/t5-base-korean-summarization](https://huggingface.co/eenzeenee/t5-base-korean-summarization)
- 해당 대회에서 수행하고자하는 Task인 요약에 대해서 많은 자료들로 사전에 Finetuning 돼 있었고 실제로 다른 T5 모델보다 좋은 성능을 보여주었습니다.

### Modeling Process

- _Write model train and test process with capture_

## 4. Result

### Leader Board
![07](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp-5/assets/42354230/68f8dcf0-86fe-4b5a-ab10-e7bb7c20aa2e)  
final_result: 40.4343  
rouge1-F1: 0.4953  
rouge2-F1: 0.3025  
rougeL-F1: 0.4153  

### Presentation
- ppt폴더 참조
- [대회회고](https://longing-artichoke-024.notion.site/Dialogue-Summarization-Upstage-Competition-f548fd094cf242e1809ad9f39b68283a?pvs=74)

## etc

### Reference
- train.csv, test.csv, dev.csv, sample_submission.csv 등의 파일은 DialogSum을 활용하여 만들어진 Upstage사의 2차 저작물임을 알려드립니다.
    - [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum): cc-by-nc-sa-4.0
- [SAMSUM Dataset](https://huggingface.co/datasets/samsum): cc-by-nc-nd-4.0
- Kobart: https://huggingface.co/digit82/kobart-summarization
- 원본 T5: https://huggingface.co/eenzeenee/t5-base-korean-summarization
- 해당 데이터셋으로 FineTuning한 T5: https://huggingface.co/dudcjs2779/dialogue-summarization-T5

