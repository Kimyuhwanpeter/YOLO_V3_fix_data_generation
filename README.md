# YOLO_V3_fix_data_generation
* xml이 아닌 오직 text를 이용해 입력을 생성
* 학습은 잘 되고 있음
* YOLO V3를 짜면서 들었던 생각: YOLO V2 보다는 훨씬 안전한 학습이 이뤄지고있음. 하지만 여전히 loss의 흐름이 불안정함 (즉 gradient가 불안정함) 또한 pre-trained 모델이 아니면 학습이 너무힘들어짐
* Darknet 말고 다른 네트워크도 사용해볼것
