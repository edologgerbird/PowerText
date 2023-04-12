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

            # render labels: possible implementation
            context['text_submitted'] = text
            context['label_hate'] = '1' if predict[0] == 1 else '0'
            context['label_private'] = '1' if predict[1] == 1 else '0'
            context['label_sexual'] = '1' if predict[2] == 1 else '0'
            context['label_imperson'] = '1' if predict[3] == 1 else '0'
            context['label_illegal'] = '1' if predict[4] == 1 else '0'
            context['label_ads'] = '1' if predict[5] == 1 else '0'
            context['label_ai'] = '1' if predict[6] == 1 else '0'

            # render text output: possible implementation
            context['displayed_text'] = shrink_text(text)
            flagged = '0'
            violations = []
            if predict[0] == 1:
                flagged = '1'
                violations.append('Hate')
            elif predict[1] == 1:
                flagged = '1'
                violations.append('Privacy')
            elif predict[2] == 1:
                flagged = '1'
                violations.append('Sexual')
            elif predict[3] == 1:
                flagged = '1'
                violations.append('Impersonation')
            elif predict[4] == 1:
                flagged = '1'
                violations.append('Illegal')
            elif predict[5] == 1:
                flagged = '1'
                violations.append('Advertisement')
            elif predict[6] == 1:
                flagged = '1'
                violations.append('AI Generated')

            if flagged == '0':
                message = 'The post passed text screening.'
            else:
                message = 'Detected '
                for i in violations:
                    message = message + i + ", "
                message = message[:-2] + " Content!"
            context['text_flagged'] = flagged
            context['text_message'] = message

    return render(request, "user_demo.html", context)

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
    output = []
    threshold = max(predict) / 2
    for i in predict:
        if i >= threshold:
            output.append(1)
        else:
            output.append(0)
    return output

def shrink_text(text):
    if len(text) >= 15:
        text = text[:6] + "..." + text[-6:]
    return text