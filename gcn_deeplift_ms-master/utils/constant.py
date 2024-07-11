def time_step(dataset,t):
    if dataset=='Chi':
        if t=='add':
            return [[0,84,0,90],[0,90,0,96],[0,96,0,102],[0,102,0,108]]
        elif t=='remove':
            return [[0,90,0,84],[0,96,0,90],[0,102,0,96],[0,108,0,102]]
        else:
            return [[78,84,80,86],[80,86,82,88],[82,88,84,90],[84,90,86,102]]
    elif dataset=='NYC':
        if t=='add':
            return [[0,78,0,80],[0,80,0,82],[0,82,0,84],[0,84,0,86]]
        elif t=='remove':
            return [[0,80,0,78],[0,82,0,80],[0,84,0,82],[0,86,0,84]]
        else:
            return [[78,84,79,85],[19,85,80,86],[80,86,81,87],[81,87,82,88]]
    elif dataset=='Zip':
        if t=='add':
            return [[0,78,0,79],[0,79,0,80],[0,80,0,81],[0,81,0,82]]
        elif t=='remove':
            return [[0,79,0,78],[0,80,0,79],[0,81,0,80],[0,82,0,81]]
        else:
            return [[338,354,340,356],[340,356,342,358],[342,358,344,360],[344,360,346,362]]
    elif dataset=='pheme' or dataset=='weibo':
        if t=='add':
            return[['edges_1','edges_2'],['edges_2','edges_3']]
        elif t=='remove':
            return [['edges_2', 'edges_1'], ['edges_3', 'edges_2']]
        else:
            return [['edges_2', 'edges_4']]

    elif dataset=='bitcoinalpha':
        if t=='add':
            return [[0,48,0,51],[0,51,0,54],[0,54,0,57],[0,57,0,60]]
        elif t=='remove':
            return [[0,51,0,48],[0,54,0,51],[0,57,0,54],[0,60,0,57]]
        else:
            return  [[24,48,26,50],[26,50,28,52],[28,52,30,54],[30,54,32,56]]
    elif dataset=='bitcoinotc':
        if t=='add':
            return [[0,48,0,50],[0,52,0,54],[0,54,0,56],[0,56,0,58]]
        elif t=='remove':
            return [[0, 50, 0, 48], [0, 52, 0, 50], [0, 54, 0, 52], [0, 56, 0, 54]]
        else:
            return [[24,48,26,50],[26,50,28,52],[28,52,30,54],[30,54,32,56]]
    elif dataset=='UCI':
        if t=='add' or t=='remove':
            return [[0,18,0,19],[0,19,0,20],[0,20,0,21],[0,21,0,22]]
        elif t=='remove':
            return [[0, 19, 0, 18], [0, 20, 0, 19], [0, 21, 0, 20], [0, 22, 0, 21]]
        else:
            return [[16,19,17,20],[17,20,18,21],[18,21,19,22],[19,22,20,23]]




def path_number(dataset,t,targetpath):
    if dataset=='Chi' or dataset=='NYC' or dataset=='Zip':
        if t=='add' or t=='both' or t=='remove':
            # return [1,2,3,4,5]
            #
            if targetpath > 1000:
                return [15, 16, 17, 18, 19]
            elif 500 < targetpath <= 1000:
                return [10, 11, 12, 13, 14]
            elif 100 < targetpath < 500:
                return [6, 7, 8, 9, 10]
            else:
                return [1,2,3,4,5]

    elif dataset=='weibo' or dataset=='pheme':
        if t == 'add' or t == 'both' or t == 'remove':
            if targetpath > 1000:
                return [15, 16, 17, 18, 19]
            elif 500 < targetpath <= 1000:
                return [10, 11, 12, 13, 14]
            elif 100 < targetpath < 500:
                return [6, 7, 8, 9, 10]
            else:
                return [1, 2, 3, 4, 5]
            # if targetpath > 1000:
            #     return [60,70,80,90,100]
            # elif 500 < targetpath <= 1000:
            #     return [10, 20, 30, 40, 50]
            # elif 100 < targetpath < 500:
            #     return [10,15,20,25,30]
            # else:
            #     return [1, 2, 3, 4, 5]
    elif dataset=='bitcoinalpha' or dataset=='bitcoinotc' or dataset=='UCI':
        if t == 'add' or t == 'both' or t == 'remove':
            if targetpath > 1000:
                return [60,70,80,90,100]
            elif 500 < targetpath <= 1000:
                return [10,20,30,40,50]
            elif 100 < targetpath < 500:
                return [10,12,14,16,18]
            else:
                return [1, 2, 3, 4, 5]

    elif dataset=='mutag':
        if t == 'add' or t == 'both' or t == 'remove':
            # return [1,2,3,4,5]
            if targetpath > 1000:
                return [10,11,12,13,14]
            elif 500 < targetpath <= 1000:
                return [6,7,8,9,10]
            elif 100 < targetpath < 500:
                return [3,4,5,6,7]
            else:
                return [1, 2, 3, 4, 5]




