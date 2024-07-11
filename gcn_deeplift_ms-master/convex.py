import numpy as np
from utils.utils_deeplift import *
from utils.evalation import metrics_graph

class convex_graph():
    def __init__(self,Hold,Hnew,W,goal,addedgelist,removeedgelist,addgoalpath,removegoalpath,feature,\
                 layernumbers,nclass,topk_pathlist,model,edges_new,edges_old,dataset,type):
        self.Hold=Hold
        self.Hnew=Hnew
        self.goal=goal
        self.W=W
        self.addedgelist=addedgelist
        self.removeedgelist=removeedgelist
        self.addgoalpath=addgoalpath
        self.removegoalpath=removegoalpath
        self.feature=feature
        self.layernumbers=layernumbers
        self.nclass=nclass
        self.topk_pathlist=topk_pathlist
        self.model=model
        self.edges_new=edges_new
        self.edges_old = edges_old
        self.dataset=dataset
        self.type=type

    def contribution_value(self):
        addgoalpath=self.addgoalpath
        removegoalpath=self.removegoalpath
        goal=self.goal
        Hold=self.Hold
        Hnew=self.Hnew
        W=self.W
        addedgelist=self.addedgelist
        removeedgelist=self.removeedgelist
        features_clear=self.feature
        layernumbers=self.layernumbers
        nclass=self.nclass
        deepliftresultma = np.zeros((len(addgoalpath) + len(removegoalpath), nclass))
        if self.type=='add' and (self.dataset=='Chi' or (self.dataset=='NYC') or (self.dataset=='Zip')):
            edges_new_tensor=ms.Tensor(self.edges_new)
            subset, edge_index, _, _ = k_hop_subgraph(
                goal, layernumbers, edges_new_tensor, relabel_nodes=True,
                num_nodes=None)
            submapping, subadj = subadj_map(subset, edge_index)
            # print('submapping',submapping)

            sub_Hold = subH(subset, submapping, Hold, layernumbers)
            # print('sub_Hold',sub_Hold)
            sub_Hnew = subH(subset, submapping, Hnew, layernumbers)
            goalnewaddpathmap, addedgelistmap, subedgesmap \
                = subpath_edge(self.addgoalpath, addedgelist, submapping, edge_index)
            # subh1 = subh1(subset, submapping, W, layernumbers)
            deepliftresultma = np.zeros((len(self.addgoalpath), nclass))
            for i in range(0, nclass):
                deepliftresultmap, deepliftmap = cal_contribution(goal, i,
                                                                                        layernumbers, goalnewaddpathmap,
                                                                                        sub_Hnew,
                                                                                        sub_Hold,
                                                                                        addedgelistmap, W,
                                                                                        sub_Hold[0])

                # print('deepliftresultmap',deepliftresultmap)

                deepliftresult = subpath_goalpath(deepliftresultmap, submapping)

                for j in range(0, len(addgoalpath)):
                    strpath = []
                    for pathindex in addgoalpath[j]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    deepliftresultma[j][i] = deepliftresult[c]
        elif self.type=='remove' and (self.dataset=='Chi' or (self.dataset=='NYC') or (self.dataset=='Zip')):
            edges_old_tensor = ms.Tensor(self.edges_old)
            subset, edge_index, _, _ = k_hop_subgraph(
                goal, layernumbers, edges_old_tensor, relabel_nodes=True,
                num_nodes=None)
            submapping, subadj = subadj_map(subset, edge_index)
            # print('submapping',submapping)

            sub_Hold =subH(subset, submapping, Hold, layernumbers)
            # print('sub_Hold',sub_Hold)
            sub_Hnew = subH(subset, submapping, Hnew, layernumbers)

            goalnewaddpathmap, removeedgelistmap, subedgesmap \
                = subpath_edge(self.removegoalpath, self.removeedgelist, submapping, edge_index)

            deepliftresultma = np.zeros((len(self.removegoalpath), nclass))
            for i in range(0, nclass):
                deepliftresultmap, deepliftmap = cal_contribution(goal, i,
                                                                                        layernumbers, goalnewaddpathmap,
                                                                                        sub_Hold,
                                                                                        sub_Hnew,
                                                                                        removeedgelistmap, W,
                                                                                        sub_Hold[0])

                # print('deepliftresultmap',deepliftresultmap)

                deepliftresult = subpath_goalpath(deepliftresultmap, submapping)
                # print('deeplift_ceshi', deepliftresult_ceshi)
                # print('i deepliftmap',i,deepliftmap)
                for j in range(0, len(removegoalpath)):
                    strpath = []
                    for pathindex in removegoalpath[j]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    deepliftresultma[j][i] = -deepliftresult[c]
        else:
            for i in range(0, nclass):
                deepliftresult_remove, deeplift_remove = cal_contribution(goal, i,
                                                                                 layernumbers,
                                                                                 removegoalpath,
                                                                                 Hold,
                                                                                 Hnew,
                                                                                 removeedgelist, W,
                                                                                 features_clear)
                deepliftresult_add, deeplift_add = cal_contribution(goal, i,
                                                                           layernumbers, addgoalpath,
                                                                           Hnew,
                                                                           Hold,
                                                                           addedgelist, W,
                                                                           features_clear)


                # deepliftresult = dict()
                # for key, value in deepliftresult_remove.items():
                #     deepliftresult[key] = -deepliftresult_remove[key]
                # for key, value in deepliftresult_add.items():
                #     deepliftresult[key] = deepliftresult_add[key]
                # print(i, sum(deepliftresult.values()))
                for j in range(0, len(removegoalpath)):
                    strpath = []
                    for pathindex in removegoalpath[j]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    deepliftresultma[j][i] = -deepliftresult_remove[c]
                for j in range(len(removegoalpath), len(removegoalpath) + len(addgoalpath)):
                    strpath = []
                    for pathindex in addgoalpath[j - len(removegoalpath)]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    deepliftresultma[j][i] = deepliftresult_add[c]
        return deepliftresultma

    def evaluate(self,select_addpaths,select_removepaths,select_addpaths_mask,select_removepaths_mask):

        length=len(select_addpaths)
        goal_logits_mask_list = []
        goal_logits_add_list = []
        for i in range(length):
            pa_add=select_addpaths_mask[i]
            pa_remove=select_removepaths_mask[i]
            edges_new = self.edges_new
            model = self.model
            goal_logits_mask = metrics_graph(model, self.feature, pa_add, pa_remove, edges_new,
                                    self.Hnew[self.layernumbers * 2 - 1],'mask',self.removeedgelist,self.addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)

            pa_add = select_addpaths[i]
            pa_remove = select_removepaths[i]


            goal_logits_add= metrics_graph(model, self.feature,  pa_add,pa_remove,self.edges_old,
                                    self.Hold[self.layernumbers * 2 - 1],'add',self.removeedgelist,self.addedgelist).cal()

            goal_logits_add_list.append(goal_logits_add)
        return goal_logits_mask_list,goal_logits_add_list

    def select_importantpath(self,deepliftresultma):
        topk_pathlist=self.topk_pathlist
        # deepliftresultma=self.contribution_value()
        H_new=np.mean(self.Hnew[self.layernumbers*2-1],axis=0)
        H_old = np.mean(self.Hold[self.layernumbers * 2 - 1], axis=0)

        number = softmax(H_new).T.dot(
            H_new - H_old)
        # print('number', number)
        # print(number)
        # print(H_new)
        # print(H_new.shape)
        number2 = 0
        for j in range(2):
            number2 = number2 + np.exp(H_new[j])
        number3 = softmax(H_old).T.dot(
            H_old - H_new)
        number4 = 0
        for j in range(2):
            number4 = number4 + np.exp(H_old[j])
        select_addpaths_mask = []
        select_removepaths_mask = []
        select_addpaths = []
        select_removepaths = []


        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            choicelist_mask = main_con_mask_graph(number3, number4, topk_path, deepliftresultma, self.nclass,
                                            H_new,
                                            H_old,
                                            self.layernumbers)

            if len(choicelist_mask) > 0 and choicelist_mask is not None:
                choiceresultdict = dict()
                for i in range(0, len(choicelist_mask)):
                    if i < len(self.removegoalpath):
                        strpath = []
                        for pathindex in self.removegoalpath[i]:
                            strpath.append(str(pathindex))
                        c = ','.join(strpath)
                        choiceresultdict[c] = choicelist_mask[i]
                    if i >= len(self.removegoalpath):
                        strpath = []
                        for pathindex in self.addgoalpath[i - len(self.removegoalpath)]:
                            strpath.append(str(pathindex))
                        c = ','.join(strpath)
                        choiceresultdict[c] = choicelist_mask[i]

                sortchoiceresultdict = sorted(choiceresultdict.items(), key=lambda item: item[1], reverse=True)
                print('mask sortchoiceresultdict',sortchoiceresultdict)
                pa_add = []
                pa_remove = []

                for i in range(0, topk_path):
                    deepliftpath = []
                    s1 = sortchoiceresultdict[i][0].split(',')
                    for j in s1:
                        deepliftpath.append(int(j))
                    if deepliftpath in self.addgoalpath:
                        pa_add.append(deepliftpath)
                    if deepliftpath in self.removegoalpath:
                        pa_remove.append(deepliftpath)
                select_addpaths_mask.append(pa_add)
                select_removepaths_mask.append(pa_remove)
                # edges_new = self.edges_new
                # model = self.model
                # goal_logits_mask = metrics_graph(model, self.feature, pa_add, pa_remove, edges_new,
                #                         self.Hnew[self.layernumbers * 2 - 1],'mask',self.removeedgelist,self.addedgelist).cal()
                # # goal_logits_mask = metrics(model, self.feature, self.goal, self.addgoalpath, self.removegoalpath, edges_new,
                # #                            self.dataset).cal()
                # goal_logits_mask_list.append(goal_logits_mask)



        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            choicelist = main_con_graph(number, number2, topk_path, deepliftresultma, self.nclass,
                                                 H_new,
                                                 H_old,
                                                 self.layernumbers)

            if choicelist.size != [] and choicelist is not None:
                choiceresultdict = dict()
                for i in range(0, len(choicelist)):
                    if i < len(self.removegoalpath):
                        strpath = []
                        for pathindex in self.removegoalpath[i]:
                            strpath.append(str(pathindex))
                        c = ','.join(strpath)
                        choiceresultdict[c] = choicelist[i]
                    if i >= len(self.removegoalpath):
                        strpath = []
                        for pathindex in self.addgoalpath[i - len(self.removegoalpath)]:
                            strpath.append(str(pathindex))
                        c = ','.join(strpath)
                        choiceresultdict[c] = choicelist[i]

                sortchoiceresultdict = sorted(choiceresultdict.items(), key=lambda item: item[1], reverse=True)
                print('add',sortchoiceresultdict)

                pa_add = []
                pa_remove = []

                for i in range(0, topk_path):
                    deepliftpath = []
                    s1 = sortchoiceresultdict[i][0].split(',')
                    for j in s1:
                        deepliftpath.append(int(j))
                    if deepliftpath in self.addgoalpath:
                        pa_add.append(deepliftpath)
                    if deepliftpath in self.removegoalpath:
                        pa_remove.append(deepliftpath)
                select_addpaths.append(pa_add)
                select_removepaths.append(pa_remove)

                # model=self.model
                # goal_logits_add= metrics_graph(model, self.feature,  pa_add,pa_remove,self.edges_old,
                #                         self.Hold[self.layernumbers * 2 - 1],'add',self.removeedgelist,self.addedgelist).cal()

                # goal_logits_add = metrics(model, self.feature, self.goal, self.removegoalpath, self.addgoalpath, self.edges_old,self.dataset).cal()
                # goal_logits_add_list.append(goal_logits_add)
        return  select_addpaths,select_removepaths,select_addpaths_mask,select_removepaths_mask





