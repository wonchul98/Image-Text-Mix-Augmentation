# 삼성전자 연계 프로젝트 - 오픈소스 VLM 한국어 문맹 깨뜨리기

본 프로젝트는 삼성전자와의 연계를 통해 오픈소스 시각언어모델(VLM)이 한국어 텍스트를 더욱 정확하게 인식할 수 있도록 하는 것을 목표로 합니다. 기존 VLM이 저자원 언어(Under-Resourced Language)인 한국어 텍스트를 잘 인식하지 못하는 문제점을 해결하기 위해, **이미지-텍스트 CutMix** 증강 기법을 개발하여 수만 장의 합성 이미지를 생성하고 모델을 학습시켰습니다.

---

## 발표 화면
- [SDC 2024 korea](https://www.sdc-korea.com/session/SN2024101800030)

<br/>

## 논문
- [논문 링크](https://github.com/wonchul98/Image-Text-Mix-Augmentation/blob/feature/cutmix-code/S11P21S006-125/Docs/Image_Text_Augmentation.pdf)

<br/>


## 프로젝트 개요

기존 시각언어모델(VLM)은 영어 등 고자원 언어에 비해 한국어 텍스트 인식에 어려움을 겪어왔습니다. 이를 극복하기 위해 **CutMix** 기법에 텍스트 이미지를 결합하는 **이미지-텍스트 CutMix**를 적용했고, 대량의 합성 이미지를 생성하여 모델을 파인튜닝(Fine-tuning)함으로써 **한국어 텍스트 인식 성능**을 크게 향상시켰습니다.

<br/>

## 모델 학습 순서도

<p align="center">
  <img src="https://github.com/user-attachments/assets/86c226b0-807a-4034-b863-fccfe35aab54" alt="모델 학습 순서도">
</p>


**이미지-텍스트 CutMix** - 텍스트 이미지를 실제 환경과 유사하게 합성하여 학습 효과 극대화  
**합성 데이터 생성** - 한국어 텍스트가 포함된 이미지 데이터를 다양하게 확보  
**VLM 파인튜닝(Fine-Tuning)** - 최신 시각언어모델(VLM)에 적용하여 성능 개선 

<br/>


## 학습 데이터셋의 유형

모델의 일반화 성능을 높이기 위해 다양한 유형의 학습 데이터를 생성했습니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b654523-17bb-40c1-bc2e-215dab180bb3" alt="학습 데이터셋 이미지">
</p>


| 유형 | 설명 |
|------|------|
| **단어 정면형 (Single-Word Frontal)** | 단일 단어가 정면에서 촬영된 이미지 |
| **다단어 정면형 (Multi-Word Frontal)** | 여러 단어로 구성된 문장이 정면에서 촬영된 이미지 |
| **단어 측면형 (Single-Word Skewed)** | 단어가 비스듬한 각도로 촬영된 이미지 |
| **다중 문장 정면형 (Multi-Sentence Frontal)** | 여러 문장이 포함된 정면 이미지 |

<br/>


## 웹 데모

아래 이미지는 **오픈소스 시각언어모델 MiniCPM v2.6**으로 구현된 웹 데모 스크린샷입니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/29a54349-484e-4179-9f0f-01913c3f3ce7" alt="웹 데모 스크린샷 1">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/66c3c624-59ee-45f2-8394-a977eed8ffbf" alt="웹 데모 스크린샷 2">
</p>

같은 사진에 대해서 학습 이전 모델 대비 한국어를 정확하게 인식하는 것을 확인할 수 있습니다. 
<br/>


## 모델 평가 방식

본 프로젝트에서는 한국어 텍스트 인식 성능을 종합적으로 평가하기 위해 다음과 같은 방법을 사용했습니다.

1. **평가 데이터셋**  
   - **Korean Image-Text Mix Dataset**: 합성 이미지 기반으로 모델의 학습 성능을 평가하기 위해 사용합니다.  
   - **Text-In-The-Wild Dataset**: 실세계에서 촬영된 이미지로 Zero-Shot 평가를 수행하여 모델의 일반화 성능을 검증합니다.

2. **평가 지표**  
   - **CER (Character Error Rate)**: 한국어 텍스트 인식 정확도를 측정하기 위한 지표입니다.  

    $$
    CER = \frac{N_{sub} + N_{del} + N_{ins}}{N}
    $$

<br/>


## 데이터셋 효용성

아래는 합성 데이터셋을 활용한 학습 전후 성능 비교 결과입니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/899b80ff-f0fa-4797-8687-beed106da408" alt="성능 비교 그래프">
</p>

- 기존 대비 **약 5배**에 달하는 성능 향상을 확인할 수 있습니다.

---
