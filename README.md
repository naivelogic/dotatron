# DOTAv2 Oriented Rotated Detecton using Detectron2

The project is for the __2021 Learning to Understand Aerial Images Challenge on DOTA dataset__ focused on training and benchmarking challenges for object detection in aerial images. 

* Training Dataset: https://captain-whu.github.io/DOTA/dataset.html


__Overview__
* __Inputs:__ DOTA iamges
* __Input Size:__ 1024 x 1024 x 3 | _dev to experiment: 608 x 608 x 3_
* __Outputs: 7 degrees of freedom (7-DOF)__ of objects: _(cx, cy, cz, l, w, h, θ)_
  * `cx, cy, cz`: the center coordinates
  * `l, w, h`: length, width, height of bounding box
  * `θ`: The heading angle in radians of the bounding box
* __Objects:__ 18 Dota-2.0 classes


Straightens and crops the image using the information of the __oriented rectangle boxes__ in the image, `width`, `height`, `center point` and `rotation degree`.