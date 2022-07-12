import pickle, os
def get_files_by_ID(ID, PATH, modality = 'MR'):
    if 'MR' in modality:
        if 'Prostate3T' in ID:
            files= os.walk(os.path.join(PATH, 'Prostate3T', 'Prostate-3T', ID))
            dir_structure= [(cd, d, f) for (cd, d, f) in files]
            IMG_PATH= dir_structure[1][0]+'/'+dir_structure[1][1][0]
            MSK_PATH= os.path.join(PATH, 'Prostate3T', 'Segmentations test', ID + '.nrrd')
        elif 'ProstateDx' in ID:
            files= os.walk(os.path.join(PATH, 'Prostate3T', 'PROSTATE-DIAGNOSIS', ID))
            dir_structure= [(cd, d, f) for (cd, d, f) in files]
            IMG_PATH= dir_structure[1][0]+'/'+dir_structure[1][1][0]
            MSK_PATH= os.path.join(PATH, 'Prostate3T', 'Segmentations training', ID + '.nrrd')
        elif 'Promise12_test' in ID:
            #Promise12 test set. Expected structure: Promise12_test_Case_[0-29]
            IMG_PATH= os.path.join(PATH, 'Promise12', 'Test', ID[len('Promise12_test')+1:] + '.mhd')
            MSK_PATH=''
        elif 'Case' in ID and 't2ax' not in ID:
            IMG_PATH= os.path.join(PATH, 'Promise12', 'Train', ID + '.mhd')
            MSK_PATH= os.path.join(PATH, 'Promise12', 'Train', ID + '_segmentation.mhd')
        elif 'Case' in ID and 't2ax' in ID:
            IMG_PATH= os.path.join(PATH, 'Harvard', 'Images', ID + '.nrrd')
            MSK_PATH= os.path.join(PATH, 'Harvard', 'Segmentations', 'Rater1', ID + '-TG-rater1.nrrd')
        elif 'Patient' in ID and len(ID)>len('Patient 1036'):
            IMG_PATH= os.path.join(PATH, 'Girona', 'GE', ID, 'T2W')
            MSK_PATH= os.path.join(PATH, 'Girona', 'GE', ID, 'GT', 'prostate')
        elif 'Patient' in ID and len(ID)<=len('Patient 1036'):
            IMG_PATH= os.path.join(PATH, 'Girona', 'Siemens', ID, 'T2W')
            MSK_PATH= os.path.join(PATH, 'Girona', 'Siemens', ID, 'GT', 'prostate')
        elif 'Salud' in ID:
            IMG_PATH= os.path.join(PATH, 'Salud_export_seg_latest/MR/', ID + '_img.nrrd')
            MSK_PATH= os.path.join(PATH, 'Salud_export_seg_latest/MR/', ID + '_msk.nrrd')
        elif 'IVO' in ID:
            path= r'D:\oscar\Prostate Images\IVO OUT'
            path_info= r'D:\oscar\SegNet Datasets\ivo_info.pkl'
            info= pickle.load(open(path_info, "rb"))
            IMG_PATH= os.path.join(path, info.loc[ID].Name, 'mr')
            MSK_PATH= os.path.join(path, info.loc[ID].Name, 'mr', 'PROSTATE.nrrd')
        else:
            raise FileNotFoundError('Image %s not found in %s'%(ID, PATH))
    elif 'US' in modality:
        if 'Salud' in ID:
            IMG_PATH= os.path.join(PATH, 'Salud', ID, 'us')
            MSK_PATH= os.path.join(PATH, 'Salud', ID, 'us', 'PROSTATE.nrrd')
        elif 'IVO' in ID:
            path= r'D:\oscar\Prostate Images\IVO OUT'
            path_info= r'D:\oscar\SegNet Datasets\ivo_info.pkl'
            info= pickle.load(open(path_info, "rb"))
            IMG_PATH= os.path.join(path, info.loc[ID].Name, 'us')
            MSK_PATH= os.path.join(path, info.loc[ID].Name, 'us', 'PROSTATE.nrrd')
        else:
            raise FileNotFoundError('Image %s not found in %s'%(ID, PATH))
    else:
        raise FileNotFoundError('Image %s not found in %s'%(ID, PATH))
    
    return IMG_PATH, MSK_PATH

segmentation_subsets= {
    'MR': 
        {'train':
            {'salud': ['Salud_11', 'Salud_12', 'Salud_13', 'Salud_14', 'Salud_16', 'Salud_17', 'Salud_19', 'Salud_20', 'Salud_22', 'Salud_23', 'Salud_24', 'Salud_27', 'Salud_29', 'Salud_3', 'Salud_30', 'Salud_31', 'Salud_33', 'Salud_34', 'Salud_35', 'Salud_36', 'Salud_37', 'Salud_38', 'Salud_39', 'Salud_4', 'Salud_40', 'Salud_41', 'Salud_43', 'Salud_44', 'Salud_45', 'Salud_47', 'Salud_5', 'Salud_52', 'Salud_54', 'Salud_56', 'Salud_57', 'Salud_6', 'Salud_60', 'Salud_61', 'Salud_62', 'Salud_63', 'Salud_64', 'Salud_66', 'Salud_68', 'Salud_7', 'Salud_70', 'Salud_71', 'Salud_72', 'Salud_76', 'Salud_77', 'Salud_78', 'Salud_79', 'Salud_80', 'Salud_81', 'Salud_82', 'Salud_83', 'Salud_84', 'Salud_86', 'Salud_88', 'Salud_9', 'Salud_90', 'Salud_91', 'Salud_93', 'Salud_94'],
             'ivo': ['IVO_0', 'IVO_10', 'IVO_100', 'IVO_101', 'IVO_104', 'IVO_106', 'IVO_107', 'IVO_109', 'IVO_110', 'IVO_111', 'IVO_112', 'IVO_113', 'IVO_114', 'IVO_115', 'IVO_116', 'IVO_118', 'IVO_119', 'IVO_120', 'IVO_122', 'IVO_123', 'IVO_126', 'IVO_127', 'IVO_128', 'IVO_129', 'IVO_13', 'IVO_130', 'IVO_131', 'IVO_133', 'IVO_134', 'IVO_136', 'IVO_139', 'IVO_14', 'IVO_140', 'IVO_141', 'IVO_142', 'IVO_143', 'IVO_145', 'IVO_146', 'IVO_147', 'IVO_15', 'IVO_151', 'IVO_152', 'IVO_153', 'IVO_154', 'IVO_155', 'IVO_157', 'IVO_158', 'IVO_159', 'IVO_16', 'IVO_160', 'IVO_161', 'IVO_162', 'IVO_163', 'IVO_164', 'IVO_165', 'IVO_166', 'IVO_167', 'IVO_169', 'IVO_17', 'IVO_170', 'IVO_171', 'IVO_174', 'IVO_176', 'IVO_180', 'IVO_181', 'IVO_182', 'IVO_183', 'IVO_185', 'IVO_186', 'IVO_187', 'IVO_189', 'IVO_190', 'IVO_191', 'IVO_192', 'IVO_193', 'IVO_195', 'IVO_196', 'IVO_199', 'IVO_2', 'IVO_20', 'IVO_200', 'IVO_201', 'IVO_203', 'IVO_204', 'IVO_205', 'IVO_206', 'IVO_207', 'IVO_208', 'IVO_209', 'IVO_21', 'IVO_211', 'IVO_213', 'IVO_214', 'IVO_215', 'IVO_216', 'IVO_217', 'IVO_218', 'IVO_219', 'IVO_22', 'IVO_220', 'IVO_221', 'IVO_223', 'IVO_224', 'IVO_227', 'IVO_228', 'IVO_229', 'IVO_23', 'IVO_230', 'IVO_232', 'IVO_236', 'IVO_237', 'IVO_238', 'IVO_240', 'IVO_241', 'IVO_242', 'IVO_243', 'IVO_245', 'IVO_246', 'IVO_250', 'IVO_252', 'IVO_253', 'IVO_254', 'IVO_255', 'IVO_256', 'IVO_257', 'IVO_259', 'IVO_26', 'IVO_260', 'IVO_261', 'IVO_262', 'IVO_263', 'IVO_264', 'IVO_267', 'IVO_268', 'IVO_269', 'IVO_270', 'IVO_271', 'IVO_274', 'IVO_275', 'IVO_278', 'IVO_28', 'IVO_281', 'IVO_282', 'IVO_283', 'IVO_284', 'IVO_285', 'IVO_286', 'IVO_287', 'IVO_29', 'IVO_3', 'IVO_30', 'IVO_31', 'IVO_32', 'IVO_34', 'IVO_35', 'IVO_36', 'IVO_37', 'IVO_38', 'IVO_4', 'IVO_41', 'IVO_42', 'IVO_45', 'IVO_46', 'IVO_47', 'IVO_49', 'IVO_5', 'IVO_52', 'IVO_56', 'IVO_57', 'IVO_59', 'IVO_60', 'IVO_62', 'IVO_63', 'IVO_64', 'IVO_65', 'IVO_68', 'IVO_69', 'IVO_70', 'IVO_72', 'IVO_73', 'IVO_74', 'IVO_78', 'IVO_79', 'IVO_8', 'IVO_82', 'IVO_83', 'IVO_85', 'IVO_86', 'IVO_88', 'IVO_89', 'IVO_9', 'IVO_90', 'IVO_92', 'IVO_93', 'IVO_94', 'IVO_97'],
             'promise12': ['Case01', 'Case02', 'Case03', 'Case04', 'Case05', 'Case06', 'Case07', 'Case10', 'Case11', 'Case12', 'Case13', 'Case15', 'Case16', 'Case17', 'Case19', 'Case20', 'Case21', 'Case22', 'Case23', 'Case24', 'Case25', 'Case26', 'Case27', 'Case28', 'Case29', 'Case30', 'Case31', 'Case33', 'Case34', 'Case36', 'Case38', 'Case39', 'Case41', 'Case47', 'Case49'],
             'prostate3t': ['Prostate3T-01-0003', 'Prostate3T-01-0004', 'Prostate3T-01-0006', 'Prostate3T-01-0012', 'Prostate3T-01-0013', 'Prostate3T-01-0020', 'Prostate3T-01-0021', 'Prostate3T-01-0022', 'Prostate3T-01-0023', 'Prostate3T-01-0024', 'Prostate3T-01-0026', 'Prostate3T-01-0028', 'Prostate3T-01-0029', 'Prostate3T-01-0030'],
             'girona': ['Patient 1304875', 'Patient 136144', 'Patient 1527375', 'Patient 1786687', 'Patient 203662', 'Patient 2107294', 'Patient 2201807', 'Patient 228550', 'Patient 2950797', 'Patient 3053998', 'Patient 384', 'Patient 410', 'Patient 416', 'Patient 428260', 'Patient 513', 'Patient 616760', 'Patient 634', 'Patient 72679', 'Patient 778', 'Patient 779031', 'Patient 784', 'Patient 799', 'Patient 804', 'Patient 836', 'Patient 996']},
         'val': 
            {'salud': ['Salud_0', 'Salud_1', 'Salud_10', 'Salud_18', 'Salud_2', 'Salud_21', 'Salud_32', 'Salud_49', 'Salud_50', 'Salud_51', 'Salud_55', 'Salud_69', 'Salud_87', 'Salud_92'],
             'ivo': ['IVO_1', 'IVO_117', 'IVO_12', 'IVO_135', 'IVO_138', 'IVO_144', 'IVO_149', 'IVO_168', 'IVO_172', 'IVO_175', 'IVO_18', 'IVO_194', 'IVO_198', 'IVO_202', 'IVO_226', 'IVO_231', 'IVO_233', 'IVO_235', 'IVO_239', 'IVO_247', 'IVO_265', 'IVO_27', 'IVO_273', 'IVO_276', 'IVO_277', 'IVO_40', 'IVO_43', 'IVO_44', 'IVO_51', 'IVO_53', 'IVO_61', 'IVO_71', 'IVO_75', 'IVO_76', 'IVO_77', 'IVO_80', 'IVO_81', 'IVO_84', 'IVO_91', 'IVO_95'],
             'promise12': ['Case09', 'Case18', 'Case35', 'Case44', 'Case45', 'Case48'],
             'prostate3t': [],
             'girona': ['Patient 1374223', 'Patient 144531', 'Patient 185708', 'Patient 782']},
         'test': 
             {'salud': ['Salud_15', 'Salud_25', 'Salud_28', 'Salud_42', 'Salud_46', 'Salud_48', 'Salud_53', 'Salud_59', 'Salud_67', 'Salud_74', 'Salud_8', 'Salud_85', 'Salud_95'],
             'ivo': ['IVO_102', 'IVO_103', 'IVO_105', 'IVO_108', 'IVO_11', 'IVO_121', 'IVO_124', 'IVO_125', 'IVO_132', 'IVO_148', 'IVO_150', 'IVO_173', 'IVO_178', 'IVO_179', 'IVO_184', 'IVO_188', 'IVO_197', 'IVO_210', 'IVO_212', 'IVO_222', 'IVO_225', 'IVO_24', 'IVO_244', 'IVO_248', 'IVO_249', 'IVO_25', 'IVO_251', 'IVO_272', 'IVO_279', 'IVO_33', 'IVO_39', 'IVO_50', 'IVO_55', 'IVO_58', 'IVO_6', 'IVO_66', 'IVO_67', 'IVO_7', 'IVO_87', 'IVO_96', 'IVO_98', 'IVO_99'],
             'promise12': ['Case00', 'Case08', 'Case14', 'Case32', 'Case42', 'Case43', 'Case46'],
             'prostate3t': ['Prostate3T-01-0010', 'Prostate3T-01-0027'],
             'girona': ['Patient 1041', 'Patient 2213046', 'Patient 387', 'Patient 59355', 'Patient 70168']}
        },
    'US': 
        {'train':
             {'salud':['Salud_0', 'Salud_1', 'Salud_10', 'Salud_12', 'Salud_14', 'Salud_15', 'Salud_16', 'Salud_18', 'Salud_19', 'Salud_21', 'Salud_22', 'Salud_24', 'Salud_27', 'Salud_29', 'Salud_3', 'Salud_30', 'Salud_32', 'Salud_33', 'Salud_35', 'Salud_36', 'Salud_37', 'Salud_38', 'Salud_4', 'Salud_40', 'Salud_41', 'Salud_42', 'Salud_43', 'Salud_44', 'Salud_45', 'Salud_46', 'Salud_47', 'Salud_49', 'Salud_5', 'Salud_51', 'Salud_54', 'Salud_57', 'Salud_59', 'Salud_6', 'Salud_60', 'Salud_62', 'Salud_64', 'Salud_65', 'Salud_66', 'Salud_69', 'Salud_7', 'Salud_74', 'Salud_76', 'Salud_77', 'Salud_79', 'Salud_81', 'Salud_82', 'Salud_85', 'Salud_87', 'Salud_9', 'Salud_90', 'Salud_91', 'Salud_92', 'Salud_94'],
              'ivo': ['IVO_1', 'IVO_100', 'IVO_104', 'IVO_105', 'IVO_106', 'IVO_107', 'IVO_109', 'IVO_110', 'IVO_115', 'IVO_116', 'IVO_117', 'IVO_118', 'IVO_119', 'IVO_120', 'IVO_124', 'IVO_125', 'IVO_127', 'IVO_128', 'IVO_129', 'IVO_130', 'IVO_131', 'IVO_134', 'IVO_135', 'IVO_138', 'IVO_139', 'IVO_141', 'IVO_142', 'IVO_144', 'IVO_145', 'IVO_15', 'IVO_150', 'IVO_152', 'IVO_153', 'IVO_154', 'IVO_163', 'IVO_166', 'IVO_168', 'IVO_169', 'IVO_171', 'IVO_173', 'IVO_180', 'IVO_182', 'IVO_184', 'IVO_185', 'IVO_186', 'IVO_188', 'IVO_189', 'IVO_192', 'IVO_194', 'IVO_199', 'IVO_2', 'IVO_200', 'IVO_204', 'IVO_205', 'IVO_208', 'IVO_209', 'IVO_21', 'IVO_210', 'IVO_212', 'IVO_222', 'IVO_225', 'IVO_231', 'IVO_237', 'IVO_239', 'IVO_24', 'IVO_241', 'IVO_249', 'IVO_25', 'IVO_255', 'IVO_256', 'IVO_257', 'IVO_260', 'IVO_262', 'IVO_265', 'IVO_266', 'IVO_268', 'IVO_269', 'IVO_27', 'IVO_270', 'IVO_273', 'IVO_275', 'IVO_277', 'IVO_278', 'IVO_283', 'IVO_286', 'IVO_33', 'IVO_34', 'IVO_35', 'IVO_36', 'IVO_42', 'IVO_43', 'IVO_44', 'IVO_46', 'IVO_51', 'IVO_54', 'IVO_55', 'IVO_57', 'IVO_58', 'IVO_6', 'IVO_60', 'IVO_63', 'IVO_64', 'IVO_65', 'IVO_66', 'IVO_68', 'IVO_71', 'IVO_78', 'IVO_85', 'IVO_93', 'IVO_96', 'IVO_98', 'IVO_99']},
         'val': 
            {'salud': ['Salud_11', 'Salud_13', 'Salud_25', 'Salud_31', 'Salud_39', 'Salud_48', 'Salud_56', 'Salud_67', 'Salud_70', 'Salud_78', 'Salud_80', 'Salud_88'],
             'ivo': ['IVO_10', 'IVO_161', 'IVO_17', 'IVO_175', 'IVO_181', 'IVO_20', 'IVO_218', 'IVO_224', 'IVO_227', 'IVO_230', 'IVO_245', 'IVO_248', 'IVO_254', 'IVO_259', 'IVO_267', 'IVO_271', 'IVO_274', 'IVO_280', 'IVO_3', 'IVO_38', 'IVO_41', 'IVO_45', 'IVO_74', 'IVO_80']},
         'test': 
            {'salud': ['Salud_17', 'Salud_20', 'Salud_23', 'Salud_28', 'Salud_34', 'Salud_61', 'Salud_68', 'Salud_71', 'Salud_72', 'Salud_8', 'Salud_83', 'Salud_95'], 
             'ivo': ['IVO_147', 'IVO_148', 'IVO_160', 'IVO_170', 'IVO_193', 'IVO_195', 'IVO_201', 'IVO_216', 'IVO_244', 'IVO_252', 'IVO_28', 'IVO_281', 'IVO_29', 'IVO_32', 'IVO_37', 'IVO_40', 'IVO_5', 'IVO_52', 'IVO_76', 'IVO_8', 'IVO_83', 'IVO_84', 'IVO_92', 'IVO_95']}
        },
    }