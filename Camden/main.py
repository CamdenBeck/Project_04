import customtkinter as ctk
import os
import tensorflow as tf
import numpy as np

# Create a customtkinter window
class Main():
    def __init__(self):
        self.tk = ctk.CTk()
        self.tk.geometry("1000x600")
        self.tk.title("Machine Learning Diabetes Prediction")
        self.tk.resizable(True, False)

        # Set the appearance mode and scaling
        ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

        # Create a frame for the main content
        self.frame = ctk.CTkScrollableFrame(self.tk)
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)
        # Add a label to the frame
        self.label = ctk.CTkLabel(self.frame, text="Enter info below:", font=("Arial", 24))
        self.label.grid(row=0, column=0, padx=20, pady=20)

        # Get a list of machine learning models from the 'Models' directory
        ml_models = [f for f in os.listdir('Models') if f.endswith('.h5')]
        ml_models = [model.replace('.h5', '') for model in ml_models]  # Remove the '.h5' extension

        # Create a combobox for selecting the model
        self.model_label = ctk.CTkLabel(self.frame, text="Select Model:")
        self.model = ctk.CTkComboBox(self.frame, values=ml_models, width=200)

        self.model_label.grid(row=0, column=2, padx=20, pady=20)
        self.model.grid(row=0, column=3, padx=20, pady=20)

        # Add Message Box fields for user direction
        self.highBP_msg = ctk.CTkLabel(self.frame, text='High Blood Pressure (0/1):', width=100)
        self.highCol_msg = ctk.CTkLabel(self.frame, text='High Cholesterol (0/1):', width=100, height=50)
        self.cholCheck_msg = ctk.CTkLabel(self.frame, text='Cholesterol Check (0/1):', width=100, height=50)
        self.bmi_msg = ctk.CTkLabel(self.frame, text='BMI (0-50):', width=100, height=50)
        self.smoker_msg = ctk.CTkLabel(self.frame, text='Smoker (0/1):', width=100, height=50)
        self.stroke_msg = ctk.CTkLabel(self.frame, text='Stroke (0/1):', width=100, height=50)
        self.heartDiseaseorAttack_msg = ctk.CTkLabel(self.frame, text='Heart Disease or Attack (0/1):', width=100, height=50)
        self.physActivity_msg = ctk.CTkLabel(self.frame, text='Physical Activity (0/1):', width=100, height=50)
        self.fruits_msg = ctk.CTkLabel(self.frame, text='Fruits (0/1):', width=100, height=50)
        self.veggies_msg = ctk.CTkLabel(self.frame, text='Vegetables (0/1):', width=100, height=50)
        self.hvyAlchoholConsum_msg = ctk.CTkLabel(self.frame, text='Heavy Alcohol Consumption (0/1):', width=100, height=50)
        self.anyHealthcare_msg = ctk.CTkLabel(self.frame, text='Any Healthcare (0/1):', width=100, height=50)
        self.noDocbcCost_msg = ctk.CTkLabel(self.frame, text='No Doctor due to Cost (0/1):', width=100, height=50)
        self.genHlth_msg = ctk.CTkLabel(self.frame, text='General Health (0-5):', width=100, height=50)
        self.mentHlth_msg = ctk.CTkLabel(self.frame, text='Mental Health (0-30):', width=100, height=50)
        self.physHlth_msg = ctk.CTkLabel(self.frame, text='Physical Health (0-30):', width=100, height=50)
        self.diffWalk_msg = ctk.CTkLabel(self.frame, text='Difficulty Walking (0/1):', width=100, height=50)
        self.gender_msg = ctk.CTkLabel(self.frame, text='Gender (0 for female/1 for male):', width=100, height=50)
        self.age_msg = ctk.CTkLabel(self.frame, text='Age (0-100):', width=100, height=50)
        self.education_msg = ctk.CTkLabel(self.frame, text='Education (0-16):', width=100, height=50)
        self.income_msg = ctk.CTkLabel(self.frame, text='Income (0-10):', width=100, height=50)

        self.highBP_msg.grid(row=1, column=0, padx=20, pady=5)
        self.highCol_msg.grid(row=2, column=0, padx=20, pady=5)
        self.cholCheck_msg.grid(row=3, column=0, padx=20, pady=5)
        self.bmi_msg.grid(row=4, column=0, padx=20, pady=5)
        self.smoker_msg.grid(row=5, column=0, padx=20, pady=5)
        self.stroke_msg.grid(row=6, column=0, padx=20, pady=5)
        self.heartDiseaseorAttack_msg.grid(row=7, column=0, padx=20, pady=5)
        self.physActivity_msg.grid(row=8, column=0, padx=20, pady=5)
        self.fruits_msg.grid(row=9, column=0, padx=20, pady=5)
        self.veggies_msg.grid(row=10, column=0, padx=20, pady=5)
        self.hvyAlchoholConsum_msg.grid(row=11, column=0, padx=20, pady=5)
        self.anyHealthcare_msg.grid(row=12, column=0, padx=20, pady=5)
        self.noDocbcCost_msg.grid(row=13, column=0, padx=20, pady=5)
        self.genHlth_msg.grid(row=14, column=0, padx=20, pady=5)
        self.mentHlth_msg.grid(row=15, column=0, padx=20, pady=5)
        self.physHlth_msg.grid(row=16, column=0, padx=20, pady=5)
        self.diffWalk_msg.grid(row=17, column=0, padx=20, pady=5)
        self.gender_msg.grid(row=18, column=0, padx=20, pady=5)
        self.age_msg.grid(row=19, column=0, padx=20, pady=5)
        self.education_msg.grid(row=20, column=0, padx=20, pady=5)
        self.income_msg.grid(row=21, column=0, padx=20, pady=5)

        # Add Entry fields for user input
        self.highBP = ctk.CTkEntry(self.frame, width=100)
        self.highCol = ctk.CTkEntry(self.frame, width=100)
        self.cholCheck = ctk.CTkEntry(self.frame, width=100)
        self.bmi = ctk.CTkEntry(self.frame, width=100)
        self.smoker = ctk.CTkEntry(self.frame, width=100)
        self.stroke = ctk.CTkEntry(self.frame, width=100)
        self.heartDiseaseorAttack = ctk.CTkEntry(self.frame, width=100)
        self.physActivity = ctk.CTkEntry(self.frame, width=100)
        self.fruits = ctk.CTkEntry(self.frame, width=100)
        self.veggies = ctk.CTkEntry(self.frame, width=100)
        self.hvyAlchoholConsum = ctk.CTkEntry(self.frame, width=100)
        self.anyHealthcare = ctk.CTkEntry(self.frame, width=100)
        self.noDocbcCost = ctk.CTkEntry(self.frame, width=100)
        self.genHlth = ctk.CTkEntry(self.frame, width=100)
        self.mentHlth = ctk.CTkEntry(self.frame, width=100)
        self.physHlth = ctk.CTkEntry(self.frame, width=100)
        self.diffWalk = ctk.CTkEntry(self.frame, width=100)
        self.gender = ctk.CTkEntry(self.frame, width=100)
        self.age = ctk.CTkEntry(self.frame, width=100)
        self.education = ctk.CTkEntry(self.frame, width=100)
        self.income = ctk.CTkEntry(self.frame, width=100)

        self.highBP.grid(row=1, column=1, padx=20, pady=5)
        self.highCol.grid(row=2, column=1, padx=20, pady=5)
        self.cholCheck.grid(row=3, column=1, padx=20, pady=5)
        self.bmi.grid(row=4, column=1, padx=20, pady=5)
        self.smoker.grid(row=5, column=1, padx=20, pady=5)
        self.stroke.grid(row=6, column=1, padx=20, pady=5)
        self.heartDiseaseorAttack.grid(row=7, column=1, padx=20, pady=5)
        self.physActivity.grid(row=8, column=1, padx=20, pady=5)
        self.fruits.grid(row=9, column=1, padx=20, pady=5)
        self.veggies.grid(row=10, column=1, padx=20, pady=5)
        self.hvyAlchoholConsum.grid(row=11, column=1, padx=20, pady=5)
        self.anyHealthcare.grid(row=12, column=1, padx=20, pady=5)
        self.noDocbcCost.grid(row=13, column=1, padx=20, pady=5)
        self.genHlth.grid(row=14, column=1, padx=20, pady=5)
        self.mentHlth.grid(row=15, column=1, padx=20, pady=5)
        self.physHlth.grid(row=16, column=1, padx=20, pady=5)
        self.diffWalk.grid(row=17, column=1, padx=20, pady=5)
        self.gender.grid(row=18, column=1, padx=20, pady=5)
        self.age.grid(row=19, column=1, padx=20, pady=5)
        self.education.grid(row=20, column=1, padx=20, pady=5)
        self.income.grid(row=21, column=1, padx=20, pady=5)

        # Add a button to submit the data
        self.submit_button = ctk.CTkButton(self.frame, text="Submit", command=self.submit_data)
        self.submit_button.grid(row=22, column=1, padx=20, pady=20)

        # Add a prediction result label
        self.result_label = ctk.CTkLabel(self.frame, text="Prediction will appear here", font=("Arial", 16))
        self.result_label.grid(row=23, column=1, columnspan=2, padx=20, pady=20)

    def submit_data(self):
        # Collect data from the entry fields
        data = {
            "HighBP": self.highBP.get(),
            "HighCol": self.highCol.get(),
            "CholCheck": self.cholCheck.get(),
            "BMI": self.bmi.get(),
            "Smoker": self.smoker.get(),
            "Stroke": self.stroke.get(),
            "HeartDiseaseOrAttack": self.heartDiseaseorAttack.get(),
            "PhysActivity": self.physActivity.get(),
            "Fruits": self.fruits.get(),
            "Veggies": self.veggies.get(),
            "HvyAlchoholConsum": self.hvyAlchoholConsum.get(),
            "AnyHealthcare": self.anyHealthcare.get(),
            "NoDocbcCost": self.noDocbcCost.get(),
            "GenHlth": self.genHlth.get(),
            "MentHlth": self.mentHlth.get(),
            "PhysHlth": self.physHlth.get(),
            "DiffWalk": self.diffWalk.get(),
            "Sex": self.gender.get(),
            "Age": self.age.get(),
            "Education": self.education.get(),
            "Income": self.income.get(),
            "Model": self.model.get()
        }
        # Here you would typically process the data, e.g., pass it to a machine learning model
        # Run the model prediction using the collected data
        # Load the selected model
        model_path = os.path.join('Models', data["Model"] + '.h5')
        model = tf.keras.models.load_model(model_path)

        # Prepare the input data for the model
        input_data = np.array([[
            float(data["HighBP"]),
            float(data["HighCol"]),
            float(data["CholCheck"]),
            float(data["BMI"]),
            float(data["Smoker"]),
            float(data["Stroke"]),
            float(data["HeartDiseaseOrAttack"]),
            float(data["PhysActivity"]),
            float(data["Fruits"]),
            float(data["Veggies"]),
            float(data["HvyAlchoholConsum"]),
            float(data["AnyHealthcare"]),
            float(data["NoDocbcCost"]),
            float(data["GenHlth"]),
            float(data["MentHlth"]),
            float(data["PhysHlth"]),
            float(data["DiffWalk"]),
            float(data["Sex"]),
            float(data["Age"]),
            float(data["Education"]),
            float(data["Income"])
        ]])

        # Make a prediction
        prediction = model.predict(input_data)

        # Display the prediction result
        if prediction <= 0.5:
            result = "It is not likely that you have diabetes"
        else:
            result = "It may be likely that you have diabetes"

        # Update the result label with the prediction
        self.result_label.configure(text=f"Prediction: {result}({prediction[0][0]:.4f}).")  

        # Add a disclaimer label
        disclaimer = """
        This is a machine learning model and not at all meant 
        to be used for a medical diagnosis.
        It is only meant for educational purposes and should not 
        be used as a substitute for professional medical advice.
        Always consult a healthcare professional for any medical concerns.
        """
        disclaimer_label = ctk.CTkLabel(self.frame, text=disclaimer, font=("Arial", 12))
        disclaimer_label.grid(row=24, column=1, columnspan=2, padx=20, pady=5)

if __name__ == '__main__':
    app = Main()
    app.tk.mainloop()  # Start the main loop of the tkinter application