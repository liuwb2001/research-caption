https://docs.google.com/document/d/1hGdy7chRpPZN53gd_Fuy1XUEhQVwyFvhME4DFdTjr9g/edit?usp=sharing

In order to train the down-stream model to make it perform better on accident detection tasks, we need an up-stream model to caption the details of the images and get the description of them. For this model, we select to use Llava. In order to improve the precision of the generated output, we gave the model a prompt to help it. We tried three kinds of prompts. The first one is like “describe the image with details.”, which only tells the model what to do. The second one is like “Describe the image with details. If there is an accident, describe it without using the word “accident””. The third kind of prompts is like “If there is an accident, describe the accident without using the word "accident" and focusing on the environment of the road, like the distance of the items. Otherwise, describe the image with details.”, which includes some hints. 

To compare these three different prompts, we set up some experiments. The model has many different versions, including the model with 7B parameters and 13B parameters. We used the model with 7B parameters to make the experiment. The model’s ‘max_new_tokens’ was set to be 1000 in order to make the output long enough to contain all the details. 

The results shown that the first kind of prompt would generate the text including the word “accident”, which couldn’t be used in the down-stream model because the down-stream model may just indicate that there is an accident in an image only based on this word. The second prompt can help the model to detect the accident and describe the image with details. However, this description just indicated that there is an accident without describe the accident itself. The third prompt can help the model describe the details and the accident itself. The description of the accident can help the down-stream model to infer whether there is an accident.

Model outputs with three different prompts:

Prompt_1: The image depicts a scene on a road with multiple vehicles. In the foreground, there is a silver SUV that appears to be in motion, with its front end slightly elevated off the ground, suggesting it may have involved in an accident. The SUV is positioned on the right side of the road, which is marked with white lane markings. Behind the SUV, there is a red car that has been involved in a collision with the silver SUV. The red car is now stationary and is positioned on the left side of the road. The front end of the red car is damaged, with visible scratches and dents. In the background, there is a white car that is also stationary, positioned behind the red car. The white car appears to be uninvolved in the incident and is waiting for the situation to clear. The road itself is a two-way street with a center divider. There are no visible pedestrians or cyclists in the image. The setting appears to be an urban environment, with buildings and other infrastructure visible in the background. There are no visible texts or distinctive branding in the image. The style of the image is a standard, non-stylized photograph, likely taken with a smartphone or a similar type of camera. The image captures a moment of an accident, with no people visible in the immediate vicinity of the incident. 

Prompt_2: The image captures a dynamic scene on a city street. Dominating the foreground is a silver SUV, its front end crumpled and slightly tilted, indicating it has been involved in a collision. The SUV is positioned on the right side of the street, facing towards the left side of the image. In the background, two cars are visible. On the left side of the street, there's a red car, its body intact. On the right side, a white car is also present, its position suggesting it might have been involved in the incident with the SUV. The street itself is lined with trees, providing a natural backdrop to this urban setting. The road is marked with white lines, typical of city streets, and there's a sidewalk visible on the right side of the image. The image does not contain any text or additional objects that can be confidently identified. The relative positions of the objects suggest a typical city street scene with vehicles and nature coexisting.

Prompt_3: In the image, there is a scene on a road with multiple vehicles. A red car is in the process of flipping over, with its wheels off the ground and its body in mid-air. The car is positioned in the center of the road, with its front facing towards the right side of the image. A silver car is approaching the red car from the left side of the image, and it appears to be in motion, as indicated by the blurred wheels. The silver car is closer to the camera than the red car, suggesting it is further down the road. On the right side of the image, there is a black car parked or stationary. The black car is positioned behind the silver car, indicating that it is further down the road than the silver car. The road itself has white lane markings, and there are trees visible on the right side of the image, suggesting that the road is lined with vegetation. The overall scene suggests a dynamic and potentially dangerous situation on the road.

