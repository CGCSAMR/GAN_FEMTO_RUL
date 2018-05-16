import os
from skimage import data
import random as rdm
import numpy as np

global ramdom_selection
global index_bt
global test

def data_pth():
    #Acceso a directorio
    path = os.getcwd()
    #Direcctorios de carpetas
    path_dir = path + '/Data'
    #Acceso sub-carpetas
    sub_folder = os.listdir(path_dir)
    # Directorioas de entrenamiento  validacion
    train_dir = path_dir + '/' + sub_folder[0]
    test_dir = path_dir + '/' + sub_folder[1]    
    #Listas de rodamientos de entrenamiento y validacion y sets
    train_list = os.listdir(train_dir)
    test_list = os.listdir(test_dir)
    #ordenamiento por nombre del rodamiento y setas
    train_list.sort()
    test_list.sort()
    #Generacion de lista de imagenes
    train_file_list = []
    test_file_list = []

    for i in train_list:
        train_set_file = os.listdir(train_dir+'/'+ i)
        train_set_file.sort()
        train_file_list.append(train_set_file)
    
    for i in test_list:
        test_set_file = os.listdir(test_dir+'/'+ i)
        test_set_file.sort()
        test_file_list.append(test_set_file)
    
    return train_dir, test_dir, train_list, test_list, train_file_list, test_file_list
#Ejemplo de imagen cargada       
#image_ex = data.load(train_dir + '/' + train_list[0] + '/' + train_file_list[0][0], as_grey = True)

def index_dir(file_list):
    #Generacion de lsitas de ordenamiento
    index_table = []  #Numero para la seleccion
    unit_table =[]    #Numero de rodameinto a probar
    number_unit = []  #Numero de ciclo de la unidad

    index = 0
    un_index = 0
    for i in file_list:
        nr_index = 0
        for j in i:
            index_table.append(index)
            unit_table.append(un_index)
            number_unit.append(nr_index)
            index += 1
            nr_index += 1
        un_index += 1
    return index_table, unit_table, number_unit   

def random_selection(index_list, batch_size,
                     index_table, unit_table, number_unit,
                     t_dir, t_list, t_file_list):
    """
    random_selection: int int list list list str list list-> array
    saca un batch de los ejemplos selecionados aleatoriamente
    """
    out = []
    if index_list == 0:
        rdm.shuffle(index_table)
    
    end_index = index_list + batch_size
    while index_list < end_index:
        example = index_table[index_list]
        un_example = unit_table[example]
        nr_example = number_unit[example]
        
        out.append(data.load(t_dir + '/' + t_list[un_example]+ '/' + t_file_list[un_example][nr_example], as_grey = True))
        index_list += 1
        if index_list == len(index_table):
            end_index = end_index - index_list
            index_list = 0
            rdm.shuffle(index_table)
            
    return index_list, np.array(out).reshape(batch_size,64,64,1)

def selection(unit_table, number_unit, t_dir, t_list, t_file_list):
    """
    selection: list list str list list-> array
    coloca todas las muestras en unarreglo para ser procesadas
    """
    out = []
    rul_out = []
    
    index = range(len(unit_table))
    size = len(unit_table)
    for i in index:
        un_example = unit_table[i]
        nr_example = number_unit[i] 
        
        bearing = t_list[un_example]
        fail_cycle = len(t_file_list[un_example])
        img_set = t_file_list[un_example][nr_example]
        
        rul_out.append(rul(bearing, img_set, fail_cycle))
        
        out.append(data.load(t_dir + '/' + bearing + '/' + img_set, as_grey = True))

    return rul_out, np.array(out).reshape(size,64,64,1)

def random_lstm(index_list, batch_size, length_lstm, rul,
                index_table, number_unit, gan_out_table,
                t_dir, t_list, t_file_list):
    """
    random_selection: int int list list list str list list-> array
    arma un batch con una serie de valores
    """
    out = []
    rul_out = []
    seq_len = []
    if index_list == 0:
        rdm.shuffle(index_table)
    
    end_index = index_list + batch_size
    
    while index_list < end_index:
        
        example = index_table[index_list]
        rul_out.append(rul[example])
        #example: indice que guia al ejemplo
        #un_example: unidad rodamiento al que partenece la muestra
        #nr_example: numero de muestra a usar

        nr_example = number_unit[example]
        
        init_data = example - nr_example
        
        if nr_example <= length_lstm:
            seq_len.append(nr_example)
        else:
            seq_len.append(length_lstm)
            
        #print(str(example) + ' ' + str(nr_example))
        #print(init_data)
        serie = []
        
        if nr_example <= length_lstm:
            end_serie = init_data + length_lstm - 1
            
            while init_data <= end_serie:
                
                if init_data <= example:
                    serie.append(gan_out_table[init_data])
                else:
                    serie.append(0*gan_out_table[example])
                init_data = init_data + 1
        else:
            lower_bound = example-length_lstm + 1
            while lower_bound <= example:
                serie.append(gan_out_table[lower_bound])
                lower_bound += 1 
                
        out.append(serie)
        index_list += 1
        
        if index_list == len(index_table):
            end_index = end_index - index_list
            index_list = 0
            rdm.shuffle(index_table)
      
    return index_list, np.array(out), np.reshape(np.array(rul_out),(len(rul_out),1)), np.array(seq_len)


def rul(bearing, img_set, fail_cycle):
    """
    bearing: str str int -> int
    Obtiene el RUL de la muestra
    """
    type_br = int(bearing[-3:-2])
    cycle = int(img_set[-13:-8])
    
    if type_br == 1:
        rul_es = 10*(fail_cycle - cycle)
    elif type_br == 2:
        rul_es = 10*(fail_cycle - cycle)
    elif type_br == 3:
        rul_es = 10*(fail_cycle - cycle)
    
    return rul_es
    
#def time_series(index, unit, number):
    """
    times_series: -> array
    obtiene una serie de valores para 
    """
def eval_data(data, number_unit, max_seq):
    out = []
    unit_pack = []
    seq_len = []
    counter = 0
    for i in range(len(data)-1):
        if number_unit[i+1] != 0:
            unit_pack.append(data[i])
            counter += 1
        else:
            seq_len.append(number_unit[i-1]+1)
            while counter < max_seq:
                unit_pack.append(0*data[i])
                counter += 1
            out.append(np.array(unit_pack))
            unit_pack = []
            counter = 0
    seq_len.append(number_unit[-1]+1)
    while counter < max_seq:
        unit_pack.append(0*data[i])
        counter += 1
    out.append(np.array(unit_pack))
    return out, seq_len