# YOLO_V3_fix_data_generation
* xml이 아닌 오직 text를 이용해 입력을 생성
* 학습은 잘 되고 있음
* YOLO V3를 짜면서 들었던 생각: YOLO V2 보다는 훨씬 안전한 학습이 이뤄지고있음. 하지만 여전히 loss의 흐름이 불안정함 (즉 gradient가 불안정함) 또한 pre-trained 모델이 아니면 학습이 너무힘들어짐. 클래스 불균형에 관한 문제가 일반적인 regression, classification 들보다 훨씬 민감하게 작용함 (focal loss를 왜 detection에서 썼는지 이해가 감)
* Darknet 말고 다른 네트워크도 사용해볼것
* 밑에 사진은 voc2007 데이터를 5에폭 이전으로 실험했을 때의 샘플 결과
<br/>


## Epoch ~ 4
![Epoch ~ 4](https://github.com/Kimyuhwanpeter/YOLO_V3_fix_data_generation/blob/main/1500_7.jpg)
<br/>

## Epoch ~ 5
![Epoch ~ 5](https://github.com/Kimyuhwanpeter/YOLO_V3_fix_data_generation/blob/main/2500_7.jpg)
