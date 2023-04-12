import random

import numpy as np
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .handler_bert import *

@csrf_exempt
def homepage(request):
    context = {}
    if request.method == 'POST':
        if request.POST['form_id'] == 'text_submit':
            text = request.POST['comment']
            predict = wrapper_hate_bert(text)
            context['text_submitted'] = text
            context['label_hate'] = '1' if predict[0] == 1 else '0'
            context['label_private'] = '1' if predict[1] == 1 else '0'
            context['label_sexual'] = '1' if predict[2] == 1 else '0'
            context['label_imperson'] = '1' if predict[3] == 1 else '0'
            context['label_illegal'] = '1' if predict[4] == 1 else '0'
            context['label_ads'] = '1' if predict[5] == 1 else '0'
            context['label_ai'] = '1' if predict[6] == 1 else '0'
    return render(request, "index.html", context)

# helper functions
def wrapper_all_zero(text):
    return np.zeros(7)

def wrapper_random_guess(text):
    output = []
    for i in range(7):
        output.append(random.choice([0, 1]))
    return output

def wrapper_hate_bert(text):
    predict = run_hate_bert(text)
    print(predict)
    output = []
    for i in predict:
        if i > 5:
            output.append(1)
        else:
            output.append(0)
    return output