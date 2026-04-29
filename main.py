#-------------------------------------------------------------------#
# Imports
#-------------------------------------------------------------------#

from secrets import token_bytes
from flask import Flask, render_template, request, session, url_for
import logging
from logging import Formatter, FileHandler
from forms import *
from werkzeug.utils import secure_filename
import os

#-----------------Keras Model Implementation-----------------------#

from keras.models import load_model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image as im



#-------------------------------------------------------------------#
# App Configs
#-------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

app.config['IMAGE UPLOADS'] = '/static/image/uploads'
# in order to use flask sessions, we need secret key :)
random_string = os.urandom(12).hex()
print("Secret key is: ", random_string)
app.secret_key = random_string




#-------------------------------------------------------------------#
# App Routes
#-------------------------------------------------------------------#
@app.route('/')
@app.route('/home')
def home():
		return render_template("forms/index.html")

@app.route('/try_now')
def try_now():
		return render_template("forms/try_now.html")

@app.route('/learn')
def learn():
		return render_template("forms/learn.html")

@app.route('/about')
def about():
		return render_template("forms/about.html")

@app.route('/copywrite')
def copywrite():
		return render_template("forms/copywrite.html")

@app.route("/q2", methods=['GET', 'POST'])
def q2():
		if request.method == 'POST':

				uploaded_image = request.files['file']
				uploaded_image_filename = uploaded_image.filename
				uploaded_image.save('static/image/uploads/' + uploaded_image_filename)
				#image = request.files['file']
				#filename = secure_filename(image.filename)
				#basedir = os.path.abspath(os.path.dirname(__file__))
				#local_filename = os.path.join(basedir, app.config['IMAGE UPLOADS'], filename)
				local_filename = 'static/image/uploads/' + uploaded_image_filename
				local_filename2 = (local_filename.split("/"))[-1]
				image4 = im.open(local_filename)
				image4.save(local_filename)

				model_path = 'SkinCancerClassificationModelhdf5nc.h5'

				# dimensions of our images
				img_width, img_height = 28, 28

				# load the model we saved
				model = load_model(model_path, compile=False)
				model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

				# load image
				img = load_img(local_filename, target_size=(img_width, img_height))
				x = img_to_array(img)
				x = np.expand_dims(x, axis=0)
				images = np.vstack([x])

				# predict image
				prediction_metrics = ((model.predict(images, batch_size=10)).tolist())[0]

				# round prediction metrics
				for i in range(len(prediction_metrics)):
						prediction_metrics[i] = (round(prediction_metrics[i] * 1000)) / 10.0

				# sort metrics in asc order
				sorted_metrics = np.argsort(np.array(prediction_metrics))

				# retrieve diagnosis
				diagnosis_class = np.argmax(prediction_metrics)



				#print(prediction_metrics)
				# -----------------------------------------KERAS MODEL IMPLEMENTATION-----------------------------------------------------

				#FOR NOW - none is <0.1%, low is 0.1-1%, medium is 1-10%, and high is 10-100%

				class_labels = {4: ('nv', ' melanocytic nevi'),
					 6: ('mel', 'melanoma'),
					 2 :('bkl', 'benign keratosis-like lesions'),
					 1:('bcc' , ' basal cell carcinoma'),
					 5: ('vasc', ' pyogenic granulomas and hemorrhage'),
					 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
					 3: ('df', 'dermatofibroma'),
					 7: ('nc', 'non-cancerous')}




				diagnosisNo = "Congratulations! Your test results show little to no risk of any skin condition."

				diagnosisNc = "Congratulations! This image is non-cancerous."

				diagnosisLow = ["low risk of Actinic Keratoses and Intraepithelial Carcinoma (also known as Bowen's Disease).",
												"low risk of Basal Cell Carcinoma.",
												"low risk of Benign Keratosis-Like Lesions.",
												"low risk of Dermatofibroma.",
												"low risk of Melanocytic Nevi.",
												"low risk of Vascular Lesions.",
												"low risk of Melanoma."]

				diagnosisMedium = ["medium risk of Actinic Keratoses and Intraepithelial Carcinoma (also known as Bowen's Disease).",
														"medium risk of Basal Cell Carcinoma.",
														"medium risk of Benign Keratosis-Like Lesions.",
														"medium risk of Dermatofibroma.",
														"medium risk of Melanocytic Nevi.",
														"medium risk of Vascular Lesions.",
														"medium risk of Melanoma."]

				diagnosisHigh = ["high risk of Actinic Keratoses and Intraepithelial Carcinoma (also known as Bowen's Disease).",
												"high risk of Basal Cell Carcinoma.",
												"high risk of Benign Keratosis-Like Lesions.",
												"high risk of Dermatofibroma.",
												"high risk of Melanocytic Nevi.",
												"high risk of Vascular Lesions.",
												"high risk of Melanoma."]



				#Percentages
				akiec = str(prediction_metrics[0])
				bcc = str(prediction_metrics[1])
				bkl = str(prediction_metrics[2])
				df = str(prediction_metrics[3])
				mel = str(prediction_metrics[6])
				nv = str(prediction_metrics[4])
				vasc = str(prediction_metrics[5])
				nc = str(prediction_metrics[7])


				description0 = """Actinic Keratoses (AK) and Intraepithelial Carcinoma, often referred to as squamous cell carcinoma in situ, are two related skin conditions that represent different stages of the same disease process. These conditions are primarily associated with prolonged sun exposure and are considered precancerous or early stages of skin cancer. Actinic keratoses, also known as solar keratoses or senile keratoses, are small, rough, scaly patches or growths that develop on the skin. They are most commonly found in areas that have been exposed to the sun over an extended period, such as the face, ears, neck, scalp, chest, backs of hands, forearms, or lips."""

				description1 = """Basal cell carcinoma (BCC) is the most common form of skin cancer, primarily affecting the basal cells, which are found in the deepest layer of the epidermis (the outer layer of the skin). This type of skin cancer is generally slow-growing and rarely spreads to other parts of the body, but it can be locally destructive if left untreated."""

				description2 = """Benign keratosis-like lesions are skin growths or skin conditions that resemble keratoses but are noncancerous (non-malignant) in nature. They are often characterized by the presence of thickened, scaly, or rough skin patches or growths. These lesions are typically not a cause for major concern, but their appearance can sometimes be bothersome or cosmetically undesirable."""

				description3 = """Dermatofibroma, also known as a cutaneous fibrous histiocytoma, is a benign skin lesion or bump that typically develops on the legs or other areas of the body. These growths are generally harmless and rarely cause medical issues, but they can persist for years or even indefinitely."""

				description4 = """Melanocytic nevi, commonly known as moles, are benign skin growths that originate from melanocytes, the pigment-producing cells in the skin. These nevi can appear virtually anywhere on the body and vary in size, shape, and color."""

				description5 = """Pyogenic granuloma, also known as lobular capillary hemangioma, is a benign vascular growth that can occur on the skin and mucous membranes. Despite the name, it is not an infection or a true granuloma but rather a proliferation of blood vessels."""

				description6 = """Melanoma is a type of skin cancer that develops from melanocytes, the pigment-producing cells in the skin. It is considered the most aggressive form of skin cancer and has the potential to spread to other parts of the body, making early detection and treatment crucial for the best prognosis."""

				description7 = "High confidence that the image does not display cancer."

				descriptions = [description0, description1, description2, description3, description4, description5, description6, description7]
				description = descriptions[diagnosis_class]

				percentage = prediction_metrics[diagnosis_class]
				diagnosis = diagnosisNo
				if diagnosis_class != 7:
						if percentage < 0.1:
								diagnosis = str(percentage) + "%: " +  diagnosisNo
						elif percentage < 1:
								diagnosis = str(percentage) + "%: " +  diagnosisLow[diagnosis_class]
						elif percentage < 10:
								diagnosis = str(percentage) + "%: " +  diagnosisMedium[diagnosis_class]
						elif percentage <= 100:
								diagnosis = str(percentage) + "%: " +  diagnosisHigh[diagnosis_class]
				else:
						diagnosis = str(percentage) + "%: " +  diagnosisNc

				return render_template("forms/q2_new.html", filename=local_filename2, diagnosis=diagnosis, akiec=akiec, bcc=bcc, bkl=bkl, df=df, mel=mel, nv=nv, vasc=vasc, nc=nc, description=description)

		elif request.method == 'GET':
				return render_template("forms/q2_new.html")

#@app.route('/q2/<filename>')
@app.route('/q2/<filename>')
def displayQ2():
		return redirect( url_for('static', filename = '/image/uploads/' + filename) , code=301)
		#return redirect( url_for('static', filename = filename) , code=301)

@app.route('/register')
def register():
		form = RegisterForm(request.form)
		return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
		form = ForgotForm(request.form)
		return render_template('forms/forgot.html', form=form)
	
# Error handlers.

@app.errorhandler(500)
def internal_error(error):
		#db_session.rollback()
		return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
		return render_template('errors/404.html'), 404

if not app.debug:
		file_handler = FileHandler('error.log')
		file_handler.setFormatter(
				Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
		)
		app.logger.setLevel(logging.INFO)
		file_handler.setLevel(logging.INFO)
		app.logger.addHandler(file_handler)
		app.logger.info('errors')

if __name__ == '__main__':
	port = int(os.environ.get('PORT', 2500))
	app.run(host='0.0.0.0', port=81)
