import glob
from shutil import copyfile
import os


from_to = {
    '1_14':'1_9',
    '1_14':'1_13',
    '6_101':'6_99',
    '7_12':'7_93',
    '7_66':'7_9',
    '7_107':'7_113',
    '7_87':'7_110',
    '7_92':'7_86',

    '8_56':'8_5',
    '9_57':'9_17',
    '9_56':'9_67',
    '10_58':'10_116',
    '10_64':'10_77',
    '10_64':'10_85',

    '11_44':'11_45',
    '11_13':'11_92',
    '11_23':'11_92',
    '11_20':'11_92',
    '12_7':'12_73',
    '12_120':'12_117',

    '14_60':'14_44',
    '14_41':'14_16',
    '14_90':'14_56',

    '15_36':'15_6',
    '15_12':'15_34',

    '16_64':'16_13',

    '19_107':'19_76',
    '19_51':'19_31',
    '19_19':'19_47',
    '19_116':'19_49',
    '20_62':'20_63',

    '21_64':'21_119',
    '21_67':'21_23',
    '21_64':'21_9',
    '21_67':'21_35',
    '21_64':'21_70',
    '21_67':'21_21',
    '22_107':'22_58',
    '22_74':'22_116',

    '23_31':'23_12',
    '23_31':'23_28',
    '24_17':'24_120',

    #removal
    '0_0':'25_89', 
    '25_120':'25_95',
    '25_120':'25_112',
    '25_17':'25_49',
    '25_13':'25_31',
    '25_10':'25_81',
    '25_10':'25_59',
    '25_44':'25_33',
    '25_117':'25_101',

    '26_18':'26_44',
    '26_29':'26_57',
    '26_2':'26_117',

    '0_0':'27_44',
    '0_0':'27_54',
    '0_0':'27_112',

    '0_0':'28_27',
    '28_31':'28_118',
    '28_19':'28_100', #p
    '28_111':'28_102',

    '29_71':'29_70',
    '29_32':'29_76',
    '29_109':'29_86',
    '29_84':'29_103',

    '30_39':'30_45',
    '30_60':'30_57',
    '30_66':'30_93',
    '30_1':'30_50',
    '30_75':'30_40',
    '30_21':'30_47',
    '30_83':'30_112',
    '30_25':'30_11',
    '30_12':'30_34',
    '30_21':'30_22',
    '30_38':'30_80',

    '31_29':'31_100',
    '31_29':'31_93',
    '31_29':'31_55',
    
    '32_113':'32_4',

    '34_52':'34_70',
    '35_28':'35_99',
    '36_117':'36_9',

    '37_74':'37_73',

    '0_0':'38_7',
    '0_0':'40_20',
    '41_90':'41_92',
    '0_0':'43_36',
    '44_5':'44_4',
    '45_50':'45_108',
    '0_0':'46_6',

    '47_37':'47_96',
    '47_41':'47_7',
    '47_76':'47_88',
    '47_3':'47_84',
    '47_46':'47_77',
    '47_56':'47_38',
    '47_108':'47_64',
    '47_3':'47_32',

    '0_0':'3_10',
    '0_0':'40_3',
    '0_0':'40_58',
    '0_0':'18_48',
    '0_0':'12_24',
    '0_0':'12_12',
    '0_0':'12_33',
    '0_0':'18_21'

    #nearest_neighbor:39_74
    #














    



    






}

if __name__ == "__main__":
    folder = '../data/train_edited/'


    for from_number, to_number in from_to.iteritems():
        fr = os.path.join(folder,from_number+'_mask.tif')
        to = os.path.join(folder,to_number+'_mask.tif')

        copyfile(fr, to)