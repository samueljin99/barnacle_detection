# DALI Developer Challenge
1. Brainstorm an idea for a system that can help the scientists get their work done faster. This only needs to be at a level of detail of a few sentences. 

	Looking at the provided sample data, there are only a few sample images to train on, compared to typical object detection problems. I assume that the real pipeline would contain more images, although still a limited amount.
	So, I need to use image augmentation techniques to expand my data size. This can be done using operations like flip, resize, crop, rotate. I can also segment the big picture into smaller grids, so the one picture becomes many pictures. This gives you more training data.
	We also need to convert the training image masks to precise position labels, in order for us to train a computer vision model. This can be done with an OpenCV contour detection as seen below

	<img width="589" alt="Screenshot 2025-01-25 at 11 41 20 PM" src="https://github.com/user-attachments/assets/d153483d-5c8e-424f-82af-ff9d43a99fdf" />


Lastly, I need to implement a model that takes in the training data to generate new predictions for the locations of barnacles.I implemented a deep learning model called YOLO that’s open source. I trained this model on my augmented data and applied it on the test, unseen data set. It looks like the performance is pretty precise, in general.

2. Identify one or more critical subtasks that are necessary to solving the task. Define these tasks clearly by thinking about what information each one will receive and what outputs they are responsible for providing. 


Data augmentation: We don’t have the amount of data needed for a computer vision model, so we have to increase our data size. Additionally, the barnacles are packed very densely. A typical object detection program that bounds the boxes would get confusing, because it’s hard to tell what’s going on, since the boxes would all be overlapping. So I split the dataset into smaller sub-images, and zoomed into each image. This serves two purposes: This way you get more training data, and the object detection labels are much clearer, since the barnacles’ positions are more distinct.

Also, We are provided the image masks, but they are all irregularly shaped and we don’t have precise quantitative information about this. So, we first need to convert the image masks into precise label data that we can feed into our model.
My solution: Using the simple OpenCV contour detection on the image masks, we convert the blue image mask drawings to precise bounding box rectangles. We got them by bounding around the masked blue ellipses. Then we’re planning to segment them and do image augmentation to get more data.
    
    Note: The next three bullets are written in a modeling/automation focused way. If you want to focus more on visualization, feel free to deviate a bit. There may not be metrics with which to evaluate, for instance. Instead, focus on whether your visualization actually gives you the insights you were hoping to obtain
    
3. Pick a subtask that you are interested in working on, and think about how you can evaluate performance on it. What metrics do you care about, and is it practical for you to compute them with just the data we’ve provided for you?

	The subtask I care about is how closely my model's individual predictions matched the placement of individual barnacles. Although the goal of the task is to count the number of barnacles, I didn’t think this was the most productive metric. This is because the model is not complete yet, so we still have future improvement in mind. If we simply spit out the number of barnacles, we have no idea which barnacle we would have missed or overcounted. This would make it really difficult to improve the model in the future. So, I cared about the collective distance from the model's prediction’s to the ground truth from the image masks.
	I didn’t ultimately compute this, but this could possibly be done by calculating the distance from the actual rectangles to my predictions, and minimize the total distance. To match up the original barnacle location to the prediction, you could use some sort of modified k-means clustering algorithm.

4. Build a prototype which attempts to do your identified subtask

See attached github repository

5. Analyze the performance of your prototype, and report your results!
    
    Overall, I was pleasantly surprised by the performance of the model. On the unseen test data, it correctly detects most of the barnacles, except for the “gaping” barnacles that open up and have a black hole in them, which I discuss in the following question.
    
6. Make some conclusions! If you built some type of automation prototype, is your approach worth pursuing? What might work better? If you build a visualization, what does your prototype tell you?

	Overall, this tells me that data augmentation is a very powerful technique worth pursuing. A simple operation can increase the total amount of data tremendously, since for composite operations, you multiply all the numbers of individual operations together. This effect was clearly seen in my model testing and improvement phase.
	The main issue was the “gaping” barnacles that open up and have a black hole in them. Perhaps I could fix this by tweaking some parameters in my model to lower the threshold of confidence for a prediction.

7. Tell us about your learning process. As stated, one of the main points of this application process is for you to learn new things. What is something you learned that excited you?

	I learned that often in data science problems, having a wealth of data may be more important than a well-fitted model. In my process of working on the project, the most drastic difference in model performance came from using data augmentation techniques to increase my training data size. Before, I’d thought that such a computer vision problem would rely on model tuning and using an advanced model. However, there are many powerful out-of-the box models, and the most essential part was to break the problem down into a sub-task of generating new data.
	Ultimately, I was impressed by how just a couple of images of training data could be cleverly adjusted into thousands of new data, eventually creating a pretty robust model.

