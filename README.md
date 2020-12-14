# Image vectorizer pipeline

Today we can easily leverage the most powerful convolutional neural networks to vectorize our images and use them for any application we want!

Here we'll create a simple pipeline to vecotrize all images in a given folder. As an example of what can be done with these vectors, we'll do a simple t-sne plot.

## Configuration

We'll use again [metaflow](https://metaflow.org/) to build a simple pipeline. All configurations are in the `config.yml`. Basically you define:

- the path to your pictures folder
- path where the file_list should be saved
- path where the vectors should be saved
- if classes should be inferred from the pictures parent folder. For instance, if a picture path is "./dresses/1.jpg", the class if dresses.
- If the feature vectors should be reduced - originally the feature vector generated has 2048, this option reduces it to 512

## Pipeline

Now having keras inside [tensorflow](https://www.tensorflow.org/) makes it really easy to build a vectorizing pipeline using pre-trained convolutional neural networks.

We'll use a [image data generator] to flow through the pictures folders. Each picture will be resized to a standard size (224x224), preprocessed using the provided network preprocessing function, passed through a Xception network without the classification layer.

At this point, we have a 2048-dim vector. One option I put in this pipeline is to reduce this vector to a 512-dim vector. This is done by averaging 4 by 4 the 2048 vector.

These vector are stored, alongside the filepaths of all images vectorized and, if chosen to infer classes, the classes.

## Docker

If docker is your thing, I'm providing a `Dockerfile` and a `docker-compose.yml` to make it easier to run. You can simple

- Adjust the `config.yml` to your use case
- run `docker-compose up`

And that's it. If you prefer running on your own environment, I'm providing the `requirements.txt` and you can simple run `python image_vectorizer_pipeline.py run` and you should be good.

## A simple usage example

I provided the `t-sne_plot.py` python script to get these vector and build a t-SNE plot. I'm using `bokeh` to run the scatter plot just because it has a really good hover function that allow me to show of a thumbnail of the respective images, making it easier to understand the similarity of vectors in a same region. An example below:

<figure style="width: 75%"  class="align-center">
<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-11-17-image_vectorizing/tsne.jpg" alt="">
<figcaption>Hovering over a point display thumbnail and image class</figcaption>
</figure>

## Final thoughts

There are a lot uses for vectorized images. Popular uses are approximate nearest neighbors search, which allows to find "similar" images, that can used for product matching or content based image retrieval. Another interesting use is classification - it is indeed what the convolutional neural net was about to do! But extracting the vector, you can save them and run different classification tasks on them, without having to pass through the expensive convolutional layers. It is an easy to do and really useful technique to run on your image data!