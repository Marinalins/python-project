from django.shortcuts import render
from django.http import HttpResponse

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def index(request):
    reassignment_count = request.GET['reassignment_count']
    reopen_count = request.GET['reopen_count']
    sys_mod_count = request.GET['sys_mod_count']

    made_sla = request.GET['made_sla']
    if made_sla == "True":
        made_sla = True
    else:
        made_sla = False
    
    category = request.GET['category']
    subcategory = request.GET['subcategory']
    u_symptom = request.GET['u_symptom']
    impact = request.GET['impact']
    urgency = request.GET['urgency']
    priority = request.GET['priority']

    knowledge = request.GET['knowledge']
    if knowledge == "True":
        knowledge = True
    else:
        knowledge = False

    u_priority_confirmation = request.GET['u_priority_confirmation']
    if u_priority_confirmation == "True":
        u_priority_confirmation = True
    else:
        u_priority_confirmation = False

    notify = request.GET['notify']

    state = request.GET['state']
    if state == "active":
        state = [ 1, 0, 0, 0, 0, 0, 0, 0, 0 ]
    if state == "evidence":
        state = [ 0, 1, 0, 0, 0, 0, 0, 0, 0 ]
    if state == "problem":
        state = [ 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    if state == "user":
        state = [ 0, 0, 0, 1, 0, 0, 0, 0, 0 ]
    if state == "vendor":
        state = [ 0, 0, 0, 0, 1, 0, 0, 0, 0 ]
    if state == "closed":
        state = [ 0, 0, 0, 0, 0, 1, 0, 0, 0 ]
    if state == "new":
        state = [ 0, 0, 0, 0, 0, 0, 1, 0, 0 ]
    if state == "resolved":
        state = [ 0, 0, 0, 0, 0, 0, 0, 1, 0 ]
    if state == "unknown":
        state = [ 0, 0, 0, 0, 0, 0, 0, 0, 1 ]

    update = request.GET['update'] #mm-dd-hh-wd (mois, jour, heure, jour de la semaine attachés)
    update = update.split('-')
    update_wd = []
    if update[3] == "00":
        update_wd = [ 1, 0, 0, 0, 0, 0, 0 ]
    if update[3] == "01":
        update_wd = [ 0, 1, 0, 0, 0, 0, 0 ]
    if update[3] == "02":
        update_wd = [ 0, 0, 2, 0, 0, 0, 0 ]
    if update[3] == "03":
        update_wd = [ 0, 0, 0, 3, 0, 0, 0 ]
    if update[3] == "04":
        update_wd = [ 0, 0, 0, 0, 4, 0, 0 ]
    if update[3] == "05":
        update_wd = [ 0, 0, 0, 0, 0, 5, 0 ]
    if update[3] == "06":
        update_wd = [ 0, 0, 0, 0, 0, 0, 6 ]
    
    contact = request.GET['contact']
    if contact == "direct":
        contact = [ 1, 0, 0, 0, 0 ]
    if contact == "email":
        contact = [ 0, 1, 0, 0, 0 ]
    if contact == "IVR":
        contact = [ 0, 0, 1, 0, 0 ]
    if contact == "phone":
        contact = [ 0, 0, 0, 1, 0 ]
    if contact == "self":
        contact = [ 0, 0, 0, 0, 1 ]

    # initialize list of lists 
    data = [[reassignment_count, reopen_count, sys_mod_count, made_sla, category,\
            subcategory, u_symptom, impact, urgency, priority, knowledge, \
            u_priority_confirmation, notify, state[0], \
            state[1], state[2], state[3], \
            state[4], state[5], state[6], state[7], state[8], \
            update[0], update[1], update[2], update_wd[0], \
            update_wd[1], update_wd[2], update_wd[3], update_wd[4], \
            update_wd[5], update_wd[6], contact[0], contact[1], \
            contact[2], contact[3], contact[4]]] 
  
    # Create the pandas DataFrame 
    df = pd.DataFrame(data, columns = ["reassignment_count", "reopen_count", "sys_mod_count", "made_sla", "category", \
                                       "subcategory", "u_symptom", "impact", "urgency", "priority", "knowledge", \
                                       "u_priority_confirmation", "notify", "state_Active", \
                                       "state_Awaiting_Evidence", "state_Awaiting_Problem", "state_Awaiting_User_Info", \
                                       "state_Awaiting_Vendor", "state_Closed", "state_New", "state_Resolved", "state_Unknown", \
                                       "update_month", "update_day", "update_hour", "update_weekday_0", \
                                       "update_weekday_1", "update_weekday_2", "update_weekday_3", "update_weekday_4", \
                                       "update_weekday_5", "update_weekday_6", "contact_Direct opening", "contact_Email", \
                                       "contact_IVR", "contact_Phone", "contact_Self service"]) 


    filename = 'predicteur_app/rf_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(df)
    return HttpResponse(result[0])