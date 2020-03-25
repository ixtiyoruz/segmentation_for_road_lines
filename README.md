# segmentation

currently it works only with culane dataset
<pre>
first download culane dataset and extract it, then download seg_label_generate github repository to convert the labels to normal image
https://github.com/XingangPan/seg_label_generate


please put the  label width as 10 px, if it is less than 10 px model will hardly learn something, more than 10px predictions will be errorous
just change this line :
https://github.com/XingangPan/seg_label_generate/blob/2cabaca76885d6167207a8e74edf2a7409e32379/src/main.cpp#L37


then run preprocess_data.py by changing inside of it accordingly

now you are ready to train it
firstly turn off distill_loss
after one epoch, if it finishes successfully turn it on and change the class_weights of background to 0.4 (if it doesnt work to 0.1, but it should work)

in order to run on video :
  1. put test.mp4 named video in the main folder of the project
  2. run python run_on_sequence.py

</pre>


