import gradio as gr
import joblib
import pandas as pd
import random


pipeline = joblib.load('pipeline.joblib')
X_test = pd.read_csv('X_test').reset_index()
y_test = pd.read_csv('y_test').reset_index()


def value_selector(X_test,y_test):
    random_index = random.choice(range(len(X_test)))
    selected_X, selected_y = X_test.loc[random_index],y_test['Target'].loc[random_index]
    return selected_X,selected_y

X_values, y_values = value_selector(X_test,y_test)
X_ex_1, y_ex_1 = value_selector(X_test,y_test)
X_ex_2, y_ex_2 = value_selector(X_test,y_test)
X_ex_3, y_ex_3 = value_selector(X_test,y_test)
X_ex_4, y_ex_4 = value_selector(X_test,y_test)
X_ex_5, y_ex_5 = value_selector(X_test,y_test)

def greet(Subject,Body,Year,Month,Day,ground_truth):
  """
  This is our main predict function
  """
  data_file = pd.DataFrame(columns={
    'Subject',
    'Body',
    'Year',
    'Month',
    'Day',
  })

  data_file = data_file.append({
    'Subject':Subject,
    'Body':Body,
    'Year':Year,
    'Month':Month,
    'Day':Day,
  },ignore_index=True)
  
  return pipeline.predict(data_file)[0]

description = (
  "This interface allow you to get predicted labels for 5 randomly selected emails "\
)

Subject = gr.inputs.Textbox(
  label = 'Subject of chosen e-mail',
  default = X_values['Subject']
)

Body = gr.inputs.Textbox(
  label = 'Body of chosen e-mail',
  default = X_values['Body']
)

Year = gr.inputs.Textbox(
  label = 'Year of chosen e-mail',
  default = str(X_values['Year'])
)

Month = gr.inputs.Textbox(
  label = 'Month of chosen e-mail',
  default = X_values['Month']
)

Day = gr.inputs.Textbox(
  label = 'Weekday of chosen e-mail',
  default = X_values['Day']
)

preds = gr.outputs.Textbox(
  label = 'Our predicted label is: '
)

ground_truth = gr.inputs.Textbox(
  label = 'Our ground_truth label is: ',
  default = y_values
)


iface = gr.Interface(
  fn=greet, 
  inputs=[Subject,Body,Year,Month,Day,ground_truth], 
  outputs=[preds],
  examples=[
    [X_ex_1['Subject'],X_ex_1['Body'],str(X_ex_1['Year']),X_ex_1['Month'],X_ex_1['Day'],y_ex_1],
    [X_ex_2['Subject'],X_ex_2['Body'],str(X_ex_2['Year']),X_ex_2['Month'],X_ex_2['Day'],y_ex_2],
    [X_ex_3['Subject'],X_ex_3['Body'],str(X_ex_3['Year']),X_ex_3['Month'],X_ex_3['Day'],y_ex_3],
    [X_ex_4['Subject'],X_ex_4['Body'],str(X_ex_4['Year']),X_ex_4['Month'],X_ex_4['Day'],y_ex_4],
    [X_ex_5['Subject'],X_ex_5['Body'],str(X_ex_5['Year']),X_ex_5['Month'],X_ex_5['Day'],y_ex_5]
  ],
  live=True,
  title = 'Predicting tags for emails',
  description = description,
  allow_flagging='never',
  theme='default'
)


iface.launch(share=False)
