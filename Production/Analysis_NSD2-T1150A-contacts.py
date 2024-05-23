from __future__ import print_function
import mdtraj as mdt
import numpy as np
import collections
from collections import defaultdict
import pandas as pd
from contact_map import ContactMap, ContactFrequency, ContactDifference, ResidueContactConcurrence, plot_concurrence
import pickle
import os
import re

# Input parameters. Needed to find and name the files.

Protein = 'NSD2'    # name of the protein. 
Variant = 'T1150A'  # WT: Wild Type or T1150A
Peptide = 'H3K36'   # name of the complexed peptide
methylation_state = 'me2'   # methylation state of the peptide
sim_time = '100ns'

number_replicates = 1  # number of replicates to analyze

# all trajectories in the target folder will be loaded, joined and superposed to one big trajectory, which is then analyzed
traj_dict = {}
for i in range(number_replicates):
    folder = ('/home/philipp/test_NSD2_T1150A/test_NSD2_T1150A/production_NSD2_{}_{}{}_{}_{}.h5'.format(Variant, Peptide, methylation_state, sim_time, i+1))
    print(folder)
    traj_dict[i+1]=mdt.load(folder)

traj_list = []
for key in traj_dict:
    traj_list.append(traj_dict[key])

traj_pre = mdt.join(traj_list,check_topology=True, discard_overlapping_frames=True)
traj = traj_pre.superpose(traj_pre,frame=0,parallel=True)
topology = traj.topology
print('Trajectories successfully loaded, joined and superposed')

# select peptide and protein
T1150A = topology.select('chainid 0 or chainid 1 and element != "H"')
protein = topology.select('chainid 0 or chainid 1 and element != "H"')
 
print('Calculating Frequency of contacts')
contacts = ContactFrequency(traj, query=T1150A, haystack=protein, cutoff=0.45) # cutoff = size of the sphere in nm used to calculate the contacts in.
x = contacts.residue_contacts.sparse_matrix.toarray()
df = pd.DataFrame(x)
 
#NSD2 has 230 residues + 15 residues peptide = 246
df = df.iloc[: , :-4] # drop last 4 columns
df = df[:-4] # drop last 4 rows
df.index = ['K991','H992','I993','K994','V995','N996','K997','P998','Y999','G1000','K1001','V1002','Q1003','I1004','Y1005','T1006','A1007','D1008','I1009','S1010','E1011','I1012','P1013','K1014','C1015','N1016','C1017','K1018','P1019','T1020','D1021','E1022','N1023','P1024','C1025','G1026','F1027','D1028','S1029','E1030','C1031','L1032','N1033','R1034','M1035','L1036','M1037','F1038','E1039','C1040','H1041','P1042','Q1043','V1044','C1045','P1046','A1047','G1048','E1049','F1050','C1051','Q1052','N1053','Q1054','C1055','I1056','F1057','T1058','K1059','R1060','Q1061','Y1062','P1063','E1064','T1065','K1066','I1067','I1068','K1069','T1070','D1071','G1072','K1073','G1074','W1075','G1076','L1077','V1078','A1079','K1080','R1081','D1082','I1083','R1084','K1085','G1086','E1087','F1088','V1089','N1090','E1091','Y1092','V1093','G1094','E1095','L1096','I1097','D1098','E1099','E1100','E1101','C1102','M1103','A1104','R1105','I1106','K1107','H1108','A1109','H1110','E1111','N1112','D1113','I1114','T1115','H1116','F1117','Y1118','M1119','L1120','T1121','I1122','D1123','K1124','D1125','R1126','I1127','I1128','D1129','A1130','G1131','P1132','K1133','G1134','N1135','Y1136','S1137','R1138','F1139','M1140','N1141','H1142','S1143','C1144','Q1145','P1146','N1147','C1148','E1149','T1150','L1151','K1152','W1153','T1154','V1155','N1156','G1157','D1158','T1159','R1160','V1161','G1162','L1163','F1164','A1165','V1166','C1167','D1168','I1169','P1170','A1171','G1172','T1173','E1174','L1175','T1176','F1177','N1178','Y1179','N1180','L1181','D1182','C1183','L1184','G1185','N1186','E1187','K1188','T1189','V1190','C1191','R1192','C1193','G1194','A1195','S1196','N1197','C1198','S1199','G1200','F1201','L1202','G1203','D1204','R1205','P1206','K1207','T1208','S1209','T1210','T1211','L1212','S1213','S1214','E1215','E1216','K1217','G1218','K1219','K1220','A29','P30','A31','T32','G33','G34','F35','K36','K37','P38','H39','R40','Y41','R42','P43']
df.columns = ['K991','H992','I993','K994','V995','N996','K997','P998','Y999','G1000','K1001','V1002','Q1003','I1004','Y1005','T1006','A1007','D1008','I1009','S1010','E1011','I1012','P1013','K1014','C1015','N1016','C1017','K1018','P1019','T1020','D1021','E1022','N1023','P1024','C1025','G1026','F1027','D1028','S1029','E1030','C1031','L1032','N1033','R1034','M1035','L1036','M1037','F1038','E1039','C1040','H1041','P1042','Q1043','V1044','C1045','P1046','A1047','G1048','E1049','F1050','C1051','Q1052','N1053','Q1054','C1055','I1056','F1057','T1058','K1059','R1060','Q1061','Y1062','P1063','E1064','T1065','K1066','I1067','I1068','K1069','T1070','D1071','G1072','K1073','G1074','W1075','G1076','L1077','V1078','A1079','K1080','R1081','D1082','I1083','R1084','K1085','G1086','E1087','F1088','V1089','N1090','E1091','Y1092','V1093','G1094','E1095','L1096','I1097','D1098','E1099','E1100','E1101','C1102','M1103','A1104','R1105','I1106','K1107','H1108','A1109','H1110','E1111','N1112','D1113','I1114','T1115','H1116','F1117','Y1118','M1119','L1120','T1121','I1122','D1123','K1124','D1125','R1126','I1127','I1128','D1129','A1130','G1131','P1132','K1133','G1134','N1135','Y1136','S1137','R1138','F1139','M1140','N1141','H1142','S1143','C1144','Q1145','P1146','N1147','C1148','E1149','T1150','L1151','K1152','W1153','T1154','V1155','N1156','G1157','D1158','T1159','R1160','V1161','G1162','L1163','F1164','A1165','V1166','C1167','D1168','I1169','P1170','A1171','G1172','T1173','E1174','L1175','T1176','F1177','N1178','Y1179','N1180','L1181','D1182','C1183','L1184','G1185','N1186','E1187','K1188','T1189','V1190','C1191','R1192','C1193','G1194','A1195','S1196','N1197','C1198','S1199','G1200','F1201','L1202','G1203','D1204','R1205','P1206','K1207','T1208','S1209','T1210','T1211','L1212','S1213','S1214','E1215','E1216','K1217','G1218','K1219','K1220','A29','P30','A31','T32','G33','G34','F35','K36','K37','P38','H39','R40','Y41','R42','P43']
 
# create new folder and safe pickle, excel sheet
new_folder = 'contacts_{}_{}_{}_{}'.format(Protein, Variant, Peptide, methylation_state)
os.system('mkdir {0}'.format(new_folder))
df.to_pickle(open(new_folder + '/' +  'df_contacts_{}_{}_{}_{}.pkl'.format(Protein, Variant, Peptide, methylation_state), 'wb'))
df.to_excel(open(new_folder + '/' +  'contacts_{}_{}_{}_{}.xlsx'.format(Protein, Variant, Peptide, methylation_state), 'wb'))