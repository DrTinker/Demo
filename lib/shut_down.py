def shutDown(dely_time=5):
    import datetime
    import os

    if dely_time <= 0:
        return

    shut_time = (datetime.datetime.now()
                 + datetime.timedelta(minutes=dely_time)).strftime("%H:%M")
    print(shut_time)
    rec = os.system(
        '''schtasks /create /tn "关机" /tr "shutdown /s" /sc once /st ''' + str(shut_time))
    print('关机时间为{}:'.format(rec))
