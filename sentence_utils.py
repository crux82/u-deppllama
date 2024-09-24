import argparse
from conllu import parse_tree, parse
import random
import traceback


random.seed(23)

parser = argparse.ArgumentParser(description='Parsing 2011 Data Generator.')

parser.add_argument('-i','--input', help='Input File Path',default=None)
parser.add_argument('-d','--decodeinput', help='Decode Input File Path',default=None)
parser.add_argument('-r','--representation', help='representation', choices = ['lct', 'grct', 'loct'], default="lct")
parser.add_argument('-m','--max_length', help='Max Length', type=int, default=100000)
parser.add_argument('-n','--max_sentences', help='Max Number of Sentences', type=int, default=10000000)
parser.add_argument('-c','--corpus', help='corpus', default="")
parser.add_argument('-p','--masked_prob', help='masked_prob', type=float, default=0)
parser.add_argument('-t','--test_files', help='test function: provide gold-standard-file prediction-file', nargs=2)
parser.add_argument('--is_labeled', action="store_true", default=False)
parser.add_argument('--simple_relations', action="store_true", default=False)
parser.add_argument('--disable_tokenization', action="store_true", default=False)


args = parser.parse_args()
input_file = args.input
max_length = args.max_length
max_sentences = args.max_sentences
decode_input_file = args.decodeinput
corpus = args.corpus
masked_prob = args.masked_prob
test_files = args.test_files
representation = args.representation

is_labeled = args.is_labeled
disable_tokenization = args.disable_tokenization
simple_relations = args.simple_relations


assert masked_prob >= 0 and masked_prob <=1

OP = '['
CP = ']'

VERBOSE=False

def print_tree(node, tab = ""):
    print(tab + str(node))

    for child in node.children:
        print_tree(child, tab=tab+"\t")

def simplify_relations(node):
    node.token["deprel"] = node.token["deprel"].split(":")[0]
    for child in node.children:
        simplify_relations(child)


def get_masked_id(N, prob):
    lista = []
    for num in range(1, N+1):
        if random.random() <= prob:
            lista.append(num)
    return lista

def tree2string_plain(tokenlist, mylist = [], masked_id=[]):
    res = ""
    for token in tokenlist:
        if isinstance(token["id"], int):
            if token["id"] in masked_id:
                res += "<mask> " 
            else:
                res += token["form"] + " "
    return res.rstrip()

# LOCT

def tree2list_loct(node, mylist = [], masked_id=[]):

        if node.token["id"] in masked_id:
            node_str = "<mask>"
        else:
            node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")

        mylist.append(OP + " " + node_str)
        for child in node.children:
            tree2list_loct(child, mylist=mylist, masked_id=masked_id)
        mylist.append(CP)


def tree2string_loct(node, masked_id=[]):
    my_list = []
    tree2list_loct(node, my_list, masked_id)
    return " ".join(my_list)


# LCT

def tree2list_lct(node, mylist = [], masked_id=[]):
    if node.token["id"] in masked_id:
        node_str = "<mask>"
    else:
        node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")

    mylist.append(OP + " " + node_str)
    is_deprel_written = False

    if len(node.children) == 0:
        mylist.append(OP + " " + node.token["deprel"] + " " + CP)

    for i, child in enumerate(node.children):        
        if child.token["id"] > node.token["id"] and not is_deprel_written:
                mylist.append(OP + " " + node.token["deprel"] + " " + CP)
                is_deprel_written = True

        tree2list_lct(child, mylist=mylist, masked_id=masked_id)

        if i == len(node.children) - 1 and not is_deprel_written:
                mylist.append(OP + " " + node.token["deprel"] + " " + CP)
                is_deprel_written = True            

    mylist.append(CP)

def tree2string_lct(node, masked_id=[]):
    my_list = []
    tree2list_lct(node, my_list, masked_id)
    return " ".join(my_list)

# GRCT

def tree2list_grct(node, mylist = [], masked_id=[]):
    node_str = node.token["deprel"]

    mylist.append(OP + " " + node_str)
    is_deprel_written = False

    if len(node.children) == 0:
        if node.token["id"] in masked_id:
            node_str = "<mask>"
        else:
            node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")
        mylist.append(OP + " " + node_str + " " + CP)

    for i, child in enumerate(node.children):        
        if child.token["id"] > node.token["id"] and not is_deprel_written:
            if node.token["id"] in masked_id:
                node_str = "<mask>"
            else:
                node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")

            mylist.append(OP + " " + node_str + " " + CP)
            is_deprel_written = True

        tree2list_grct(child, mylist=mylist, masked_id=masked_id)

        if i == len(node.children) - 1 and not is_deprel_written:
            if node.token["id"] in masked_id:
                node_str = "<mask>"
            else:
                node_str = node.token["form"].replace("[", "-LRB-").replace("]", "-RRB-")
            
            mylist.append(OP + " " + node_str + " " + CP)
            is_deprel_written = True            

    mylist.append(CP)

def tree2string_grct(node, masked_id=[]):
    my_list = []
    #print_tree(node)
    tree2list_grct(node, my_list, masked_id)
    return " ".join(my_list)


#file = open(input_file, "r", encoding="utf8")
#content = file.readlines()


#=================DECODER=======================

# These are general decoder methods

def toTree(expression):
    tree = dict()
    msg =""
    stack = list()
    for char in expression:
        if(char == OP):
            stack.append(msg)
            msg = ""
        elif char == CP:
            parent = stack.pop()
            if parent not in tree:
                tree[parent] = list()
            tree[parent].append(msg)
            msg = parent
        else:
            msg += char
    return tree

def parseExpression(expression):
    nodeMap = dict()
    counter = 1
    node = ""
    retExp =""
    for char in expression:
        if char == OP or char == CP :
            if (len(node) > 0):
                nodeMap[str(counter)] = node;
                retExp += str(counter)
                counter +=1
            retExp += char
            node =""
        elif char == ' ': continue
        else :
            node += char
    return retExp,nodeMap


def printTree(tree, node, nodeMap):
    if node not in tree:
        return 
    print('%s -> %s' % (nodeMap[node], ' '.join(nodeMap[child] for child in tree[node]))) 
    for child in tree[node]:
        printTree(tree, child, nodeMap)

def _decode(tree, representation_type, node, nodeMap, parent, grand_parent, tid2treenodeMap, res):
    if node not in tree:
        tid = 1
        if res:
            tid = int(max(res.keys())) + 1

        grand_parent_label = "ROOT"
        if grand_parent in nodeMap:
            grand_parent_label = nodeMap[grand_parent]

        if representation_type == "lct":
            res[tid] = { "id": tid, "form": nodeMap[parent], "to": grand_parent_label, "toid" : grand_parent, "deprel": nodeMap[node] }
        elif representation_type == "grct":
            res[tid] = { "id": tid, "form": nodeMap[node], "to": grand_parent_label, "toid" : grand_parent, "deprel": nodeMap[parent] }
        else:
            raise Exception("The representation_type\t" + representation_type + "\t is not supported in decoding.")
        
        if VERBOSE:
            print(res[tid])

        tid2treenodeMap[parent] = str(tid)
        
        return 
    
    for child in tree[node]:
        _decode(tree, representation_type, child, nodeMap, node, parent, tid2treenodeMap, res) 

def decode(tree, nodeMap, representation_type="lct"):
    res = dict()
    tid2treenodeMap = dict()
    #print(tree[''][0])
    _decode(tree, representation_type, "1", nodeMap, None, None, tid2treenodeMap, res)

    for i in range(1, len(res)+1):
        if res[i]["toid"] is None:
            res[i]["toid"] = 0
        else:
            try:
                res[i]["toid"] = tid2treenodeMap[res[i]["toid"]]
            except:
                res[i]["toid"] = 0

    return res

# END of general purpose decoder methods

def check_inconsistencies(gold_tree, pred_tree):
    if len(gold_tree) != len(pred_tree):
        if abs(len(gold_tree) - len(pred_tree)) > 5:
            raise Exception("There is something strange: the difference of the trees in terms of length is too much.")
        if len(pred_tree) > len(gold_tree):
            raise Exception("There is something strange: the predicted sentence should not be longer than the original one.")    

        return True

    for i in range(1, len(gold_tree)+1):
        if gold_tree[i]["form"] != pred_tree[i]["form"]:
            #print(gold_tree[i]["form"])
            return True
    return False


def solve_inconsistencies(gold_tree, pred_tree):
    res = dict()

    if VERBOSE:
        print(gold_tree)
        print()
        print(pred_tree)

    new_id_map = dict()
    old_ids_cache = dict()

    for i in range(1, min(len(gold_tree)+1, len(pred_tree)+1)):
        old_id = pred_tree[i]["id"]
        old_toid = int(pred_tree[i]["toid"])
        old_ids_cache[i] = old_toid

        if VERBOSE:
            print("ID\t" + str(i) + "\t" + str(pred_tree[i]["form"]))
        
        if gold_tree[i]["form"] == pred_tree[i]["form"]:
            res[i] = pred_tree[i]

            new_id_map[old_id] = old_id
        else:
            if VERBOSE:
                print("Problems\t" + str(i))
            found_token = False

            #find the word in the gold sentence
            for v in range(1, 200):
                for sign in [1, -1]:
                    j = v * sign

                    if (i + j) >= 1 and (i + j) <= len(gold_tree) and gold_tree[i + j]["form"] == pred_tree[i]["form"]:
                        old_id = pred_tree[i]["id"]
                        old_form = pred_tree[i]["form"]
                        old_rel = pred_tree[i]["deprel"]

                        if old_toid < 0:
                            found_token = False                        
                            break

                        if old_toid == 0: #root
                            old_to_form = "root"
                        else:
                            old_to_form = pred_tree[int(old_toid)]["form"]
                        
                        new_id = gold_tree[i + j]["id"]
                        if VERBOSE:
                            print("ID_CHANGED\t" + str(old_id) + "\t" + str(new_id))
                        found_token = True                        
                        break
                if found_token:
                    break

            if found_token:
                #old_toid will be changed later
                new_token = { "id": new_id, "form": old_form, "to": gold_tree[new_id]["form"], "toid" : int(old_toid), "deprel": old_rel }
                res[new_id] = new_token
                new_id_map[old_id] = new_id

    # ADD MISSING TOKENS
    for i in range(1, len(gold_tree)+1):
        if i not in res:
            new_deprel = "UNK"
            if gold_tree[i]["form"] == "," or gold_tree[i]["form"] == "\"":
                new_deprel = "punct"
            res[i] = { "id": i, "form": gold_tree[i]["form"], "toid" : -1, "deprel": new_deprel }   

        if VERBOSE:
            print("Check" + str(i) + "\tfrom\t" + str(old_toid))
        old_toid = int(res[i]["toid"])
        new_toid = 0 
        if old_toid in new_id_map:
            new_toid = new_id_map[old_toid]
            if VERBOSE:
                print("Swapping" + str(i) + "\tfrom\t" + str(old_toid) + "\t"+ str(new_toid))
            res[i]["toid"] = new_toid
        else:
            if res[i]["deprel"] == "root":
                res[i]["toid"] = 0
            elif i in old_ids_cache:
                res[i]["toid"] = old_ids_cache[i] 
            else:
                res[i]["toid"] = 0
    if VERBOSE:
        print("new_id_map\t" + str(new_id_map))

    return res

#------------


# LCT

def get_decode(tree_string, representation_type):
    dep_array = []

    tree_string2, nodeMap = parseExpression(tree_string)
    tree = toTree(tree_string2)

    res = decode(tree, nodeMap, representation_type)

    return res

def print_line(line):
    return str(line["id"])+"\t"+line["form"]+"\t_\t_\t_\t_\t"+str(line["toid"])+"\t"+ line["deprel"]+"\t_\t_"


def print_decode(tree_string, representation_type):
    dep_array = get_decode(tree_string, representation_type)

    for i in range(1, len(res)+1):
        dep_array.append(print_line(res[i+1]))

    for i in range(len(dep_array)):
        print(dep_array[i])
    print()



#=================MAIN=======================        


if input_file:
    with open(input_file, 'r') as file:
        content = file.read()

        trees = parse_tree(content)

        if simple_relations:
            for tree in trees:
                simplify_relations(tree)

        sentences = parse(content)

        for i in range( min(len(trees), max_sentences)):
            #if i != 135:
            #    continue

            masked_id = get_masked_id(len(sentences[i]), masked_prob)

            if disable_tokenization:
                str_input = sentences[i].metadata["text"]
            else:
                str_input = tree2string_plain(sentences[i], masked_id=masked_id)
                
            output_loct = tree2string_loct(trees[i], masked_id=masked_id)
            output_grct = tree2string_grct(trees[i], masked_id=masked_id)
            output_lct = tree2string_lct(trees[i], masked_id=masked_id)

            if representation == "lct":
                used_representation = output_lct
            elif representation == "grct":
                used_representation = output_grct
            else:
                used_representation = output_loct

            used_representation = used_representation.replace(" ", "")

            print(corpus + "___" + str(i) + "\tparse\t" + str_input + "\t" + used_representation)

            
elif decode_input_file:
    with open(decode_input_file, 'r') as file:
        content = file.readlines()

        for line in content: 
            split = line.split("\t")
            try:
                tree_str = split[3].rstrip()

                if representation == "lct":
                    print_decode_lct(tree_str)
                elif representation == "grct":
                    print_decode_grct(tree_str)
                else:
                    raise Exception("Sorry, cannot decode loct representation")
                
            except Exception:
                print("Problems with row " + split[0] + "\t" + tree_str)

    
elif test_files:
    #print(test_files)

    gold_standard_file = test_files[0]
    prediction_file = test_files[1]

    corr = 0
    tot = 0

    with open(gold_standard_file, 'r') as file1:
        gold_standard_lines = file1.readlines()

    with open(prediction_file, 'r') as file2:
        prediction_lines = file2.readlines()

    if corpus:
        gold_standard_lines = [s for s in gold_standard_lines if s.startswith(corpus)]
        prediction_lines = [s for s in prediction_lines if s.startswith(corpus)]

    for i in range(len(gold_standard_lines)):
        try:
            gold_standard_line = gold_standard_lines[i]
            prediction_line = prediction_lines[i]
        except:
            print("ERROR: files have different Length")
            print("Gold standard file:\t" + str(len(gold_standard_lines)))
            print("Prediction  file:\t" + str(len(prediction_lines)))
            quit()

        #if "Danish-DDT___155\t" not in gold_standard_line:
        #    continue

        gold_tree = gold_standard_line.split("\t")[3].rstrip()
        pred_tree = prediction_line.split("\t")[3].rstrip()


        try:            
            gold_deps = get_decode(gold_tree, representation)
        except:
            print("Problems with gold_standard line " + gold_standard_line)
            #traceback.print_exception(*sys.exc_info())
            continue
            #

        try:            
            pred_deps = get_decode(pred_tree, representation)
        except Exception:
            try:            
                pred_deps = get_decode(pred_tree[:-1], representation)
            except Exception:
                print("Problems with prediction line " + prediction_line)
                #print(traceback.format_exc())
                continue
                #traceback.print_exception(*sys.exc_info())

        try:    
            if check_inconsistencies(gold_deps, pred_deps):
                if not pred_tree.endswith("]"):
                    print("Problems with not closed prediction " + pred_tree)
                else:
                    print("SOLVE")
                    pred_deps = solve_inconsistencies(gold_deps, pred_deps)
        except Exception:
            #print(traceback.format_exc())
            print("Problems with not too short/long prediction " + prediction_line)
            continue

        for j in range(min(len(gold_deps), len(pred_deps))):
            if not is_labeled:
                gold_rel = int(gold_deps[j+1]["toid"])
                pred_rel = int(pred_deps[j+1]["toid"])
            else:
                gold_rel = str(gold_deps[j+1]["toid"]) + "_" + str(gold_deps[j+1]["deprel"]).split(":")[0]
                pred_rel = str(pred_deps[j+1]["toid"]) + "_" + str(pred_deps[j+1]["deprel"]).split(":")[0]
            prefix = "[_]"

            if gold_rel == pred_rel:
                corr = corr + 1
                prefix = "[X]"
            tot = tot + 1
            print(prefix + "\t" + str.format('{0:.4f}',(corr/tot)) +  "\t" + print_line(gold_deps[j+1]) + "\t" + print_line(pred_deps[j+1]))
        print()

    #print(len(gold_standard_lines))
    #print(len(prediction_lines))

else:

    #===================MORE CODE===================

    print("ERROR")

    quit()

    tree_string = "[arrivare[Loro[nsubj]][non[advmod]][ci[expl]][sarebbero[aux]][mai[advmod]][potuti[aux]][root][,[punct]][seguiva[perché[mark]][mente[la[det]][loro[det:poss]][nsubj]][advcl][strade[le[det]][obj][consuete[amod]]][lottava[,[punct]][e[cc]][altra[l'[det]][nsubj]][no[advmod]][,[punct]][lottava[altra[l'[det]][nsubj]][conj][impossibile[contro[case]][l'[det]][obl]]]]][.[punct]]]"
    tree_string = "[GOVERNO[IL[det]][root][MONTI[nmod][-[punct]][NAPOLITANO[flat:name]]][:[punct]][GOVERNO[IL[det]][appos][LOGGE[DI[case]][LE[det]][nmod]][GOVERNO[,[punct]][IL[det]][conj][LACRIME[nmod][SANGUE[E[cc]][conj]]][POPOLI[PER[case]][I[det]][nmod][LAVORATORI[ED[cc]][I[det]][conj]]]][,[...[punct]]]][http://t.co/1OBUAFzO[dep]]]"
    tree_string = "[atleta[root][nasconde[che[nsubj]][non[advmod]][acl:relcl][sforzo[lo[det]][obj]][rende[ma[cc]][lo[obj]][conj][tanto[advmod]][naturale[xcomp]][sembra[che[mark]][rimonta[la[det]][sua[det:poss]][nsubj][,[punct]]][crisi[dopo[case]][una[det]][obl][chilometro[a[case]][il[det]][quindicesimo[amod]][nmod]][,[punct]]][advcl][scritta[xcomp][occhi[in[case]][gli[det]][obl]][,[punct]][tettoie[sotto[case]][le[det]][due[nummod]][obl][sopraciglia[di[case]][nmod][scure[amod]][fanno[che[nsubj]][lo[obj]][acl:relcl][uomo[più[advmod]][xcomp][quanto[di[case]][obl][Michele[non[advmod]][sia[cop]][,[punct]][avvezzo[più[advmod]][advcl][emozioni[a[case]][le[det]][obl][quanto[di[case]][nmod][sembri[non[advmod]][acl:relcl]]]]]]]]]]]]]][.[punct]]]"
    tree_string = "[cambiato[sperimentazione[La[det]][nsubj][atomica[di[case]][l'[det]][nmod]]][ha[aux]][root][mondo[il[det]][obj]][sempre[per[case]][advmod]][,[punct]][disse[si[expl]][parataxis][Hiroshima[dopo[case]][obl]]][.[punct]]]"
    tree_string = "[riguardato[diminuzione[parte[A[case]][obl][computer[personal[amod]][nmod][programmi[e[cc]][conj][composizione[di[case]][nmod]]]][,[punct]]]la[det]][nsubj][particolare[in[case]][obl]][ha[aux]][root][stampanti[le[det]][obj]][costi[-LRB-[punct]][grazie[case][a[fixed]]][i[det]][minori[amod]][obl][memorie[di[case]][le[det]][nmod][microprocessori[e[cc]][di[case]][i[det]][conj]]][-RRB-[punct]]][,[punct]][quelle[soprattutto[advmod]][fra[case]][obl][consentono[che[nsubj]][acl:relcl][gestire[di[mark]][xcomp][testo[obj][grafica[e[cc]][conj][risoluzione[ad[case]][alta[amod]][nmod]]]]]]][.[punct]]]"
    tree_string = "[root[nsubj[Parma]][conquista][obj[det[il]][premio][nmod[case[da]][nummod[200]][milioni]]][punct[.]]]"
    tree_string = "[root[obl[case[In]][det[il]][corso][nmod[case[di]][det[il]][amod[diciottesimo][conj[cc[e]][diciannovesimo]]][secolo]][punct[,]]][nsubj[det[la]][det:poss[sua]][reputazione]][expl[si]][aux[è]][diffusa][obl[advmod[anche]][case[a]][det[l']][estero]][punct[.]]]"
    tree_string2 = "[Evacuata[root][Tate[la[det]][obj][Gallery[flat:name]]][.[punct]]]"
    tree_string3 = "[conservatorio[Un[det]][ottimo[amod]][root][musicista[un[det]][valido[amod]][conj]][.[punct]]]"


    res = print_decode_grct(tree_string)

