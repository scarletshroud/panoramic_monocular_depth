# Panoramic Monocular Depth
## Приложение для построения карты глубины сцены по изображениям, снятых с различных ракурсов

Приложение работает на базе фреймворка MiDaS и поддерживает модели в формате PyTorch и оптимизированные модели в формате Onnx.  

## Использование приложения
Обученные модели в форматах .pt/.onnx помещаются в папку Weights.  
Изображения, которые следует обработать необходимо поместить в папку Img.
Результат работы будет представлен в папке Output.
Работа с приложением осуществляется с помощью командной строки, приложение принимает следующие аргументы:  
- --input - путь к папке, содержащей изображения
- --output - путь к папке для сохранения результата
- --backend - используемый фреймворк машинного обучения PyTorch/Onnx
- --model - название использумой модели

## Модели
Квантизированные модели доступны для скачивания по ссылке - https://drive.google.com/drive/folders/1qI7ZmDV44D_Y-_CrpmGeNP-5w5zOsrHs?usp=drive_link
