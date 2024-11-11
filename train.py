import numpy as np
import parking_model as pm
from tqdm import tqdm

num_prototypes = 100

def create_prototypes(param_fiz):
    prototypes = []
    for _ in range(num_prototypes):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        alpha = np.random.uniform(-np.pi, np.pi)
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        speed = np.random.choice([param_fiz.Vmod, -param_fiz.Vmod, 0]) 
        prototypes.append(np.array([x, y, alpha, angle, speed]))
    return np.array(prototypes)

def compute_similarity(state_action, prototype, sigma=1.0):
    distance = np.linalg.norm(state_action - prototype)
    similarity = np.exp(-distance ** 2 / (2 * sigma ** 2))
    print(similarity)
    return similarity

# Create the feature vector based on prototype similarities
def get_prototype_features(state_action):
    features = np.zeros(num_prototypes)
    prototypes = create_prototypes(pm.GlobalVar())
    for i, prototype in enumerate(prototypes):
        features[i] = compute_similarity(state_action, prototype)
    return features

# przykładowa nagroda za krok - nie wiem czy dobra
def nagroda_za_krok(param_fiz, stan, czy_kolizja, czy_zatrzymanie):
    # tutaj należy ustalić nagrodę za każdy krok, tak by uczenie podążało
    # we właściwym kierunku:
    # ........................................
    # ........................................

    # przykładowe obliczenie nagrody za krok - nie wiem czy dobre:
    wartosc = 0
    x = stan[0]
    y = stan[1] 
    alfa = stan[2]
    odl_xy_kw = x*x + y*y
    alfa_zred = 0
    if param_fiz.if_side_parking_place:
        if np.abs(alfa) > np.pi / 2:
            alfa_zred = np.pi - np.abs(alfa)
        else:
            alfa_zred = np.abs(alfa)
    else:
        alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)

    alfa_zred = alfa_zred/(odl_xy_kw+0.5)

    ocena_odl = 1/(odl_xy_kw+0.5)-1
    # modyfikacja nagrody:
    ocena_alfa = alfa_zred - 0.5

    # jeśli V==0 nagroda na podstawie odległości
    
    if czy_kolizja:
        wartosc = -1
    elif czy_zatrzymanie:
        # modyfikacja nagrody:
        wartosc = min(ocena_odl,ocena_alfa)
    else:
        wartosc = 0
    
    return wartosc, odl_xy_kw

def choose_action(param_fiz, stan, w):
    best_value = -float("inf")
    best_action = None
    for kat in np.linspace(-np.pi / 4, np.pi / 4, 7):
        for V in [param_fiz.Vmod, -param_fiz.Vmod, 0]:
            state_action = np.append(stan, [kat, V])
            features = get_prototype_features(state_action)
            q_value = np.dot(w, features)
            if q_value > best_value:
                best_value = q_value
                best_action = (kat, V, V == 0)
    kat, V, czy_zatrzymanie = best_action
    return kat, V, czy_zatrzymanie

# test parkowania - nie wolno niczego zmieniać!
def park_test(param_fiz, stany_poczatkowe, model, nazwa_pliku):
    pm.park_save("param.txt", param_fiz)
    phist = open(nazwa_pliku, 'w')
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape
    sr_ocena_koncowa = 0
    sr_liczba_krokow = 0 
    for epizod in range(liczba_stanow_poczatkowych):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod %  liczba_stanow_poczatkowych
        stan = stany_poczatkowe[nr_stanup,:]

        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        while czy_zatrzymanie == False:
            krok = krok + 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan zgodnie z wyuczoną strategią:
            kat, V, czy_zatrzymanie = choose_action(param_fiz,stan,model)
            
            # zapis kroku historii:
            #phist.write(str(epizod + 1) + "  " + str(krok) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(kat) + "  " + str(V) + "\n")
            phist.write("%d %d %.4f %.4f %.4f %.4f %.4f\n" % ((epizod + 1),krok,stan[0],stan[1],stan[2],kat,V))
            # wyznaczenie nowego stanu:
            nowystan, sr_obrotu, czy_kolizja = pm.model_of_car(param_fiz, stan, kat, V)

            if (czy_kolizja)|(krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            stan = nowystan
        ocena_koncowa = pm.final_score(param_fiz, nowystan, czy_kolizja, krok)
        sr_ocena_koncowa += ocena_koncowa / liczba_stanow_poczatkowych
        sr_liczba_krokow = sr_liczba_krokow + krok / liczba_stanow_poczatkowych
        print("w %d epizodzie ocena parkowania = %g, liczba krokow = %d" %(epizod, ocena_koncowa, krok))

    print("srednia ocena końcowa na epizod = %g" % (sr_ocena_koncowa))
    print("srednia liczba krokow = %g" % (sr_liczba_krokow))
    phist.close()
    return sr_ocena_koncowa


def park_train():
    liczba_epizodow = 2000
    alfa = 0.01  # wsp.szybkosci uczenia(moze byc funkcja czasu)
    epsylon = 0.1 # wsp.eksploracji(moze byc funkcja czasu)
 
    stany_poczatkowe_1 = np.array([[9.1, 4.6, 0],[6.3, 5.06, 0],[9.6, 3.15, 0],[7.3, 5.75, 0],\
                                 [10.1, 6.21, 0]],dtype=float)    # z prawej przodem w prawo
    stany_poczatkowe_2 = np.array([[9.1, 4.6, np.pi],[6.3, 5.06, np.pi],[9.6, 3.15, np.pi],\
                                 [7.3, 5.75, np.pi],[10.1, 6.21, np.pi]],dtype=float)       # z prawej przodem w lewo
    stany_poczatkowe_3 = np.array([[-9.1, 4.6, 0],[-6.3, 5.06, 0],[-9.6, 3.15, 0],[-7.3, 5.75, 0],\
                                 [-10.1, 6.21, 0]],dtype=float)    # z lewej przodem w prawo
    stany_poczatkowe_4 = np.array([[-9.1, 4.6, np.pi],[-6.3, 5.06, np.pi],[-9.6, 3.15, np.pi],\
                                 [-7.3, 5.75, np.pi],[-10.1, 6.21, np.pi]],dtype=float)       # z lewej przodem w lewo
    stany_poczatkowe = stany_poczatkowe_1
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape

    param_fiz = pm.GlobalVar()     # parametry fizyczne parkingu i pojazdu

    # inicjacja kodowania, wyznaczenie liczby parametrów (wag):
    # ........................................................
    # ........................................................

    # inicjacja wektora wag:
    w = np.zeros(num_prototypes)  # Weights for prototypes


    for epizod in tqdm(range(liczba_epizodow)):
        # epsylon = max(0.1, epsylon*0.995)
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod %  liczba_stanow_poczatkowych
        stan = stany_poczatkowe[nr_stanup, :]

        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        while not czy_zatrzymanie:
            krok = krok + 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan z uwzględnieniem
            # eksploracji (np. metoda epsylon-zachlanna lub softmax lub jeszcze inna)

            if np.random.rand() < epsylon:
                kat = np.random.uniform(-np.pi / 4, np.pi / 4)  # Random angle
                V = np.random.uniform(0, param_fiz.Vmod)  # Random speed
                czy_zatrzymanie = False 
            else:
                kat, V, czy_zatrzymanie = choose_action(param_fiz, stan, w)

            # wyznaczenie nowego stanu:
            nowystan, sr_obrotu, czy_kolizja = pm.model_of_car(param_fiz, stan, kat, V)

            if (czy_kolizja)|(krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            R, odl = nagroda_za_krok(param_fiz, nowystan, czy_kolizja, czy_zatrzymanie)
            # min_odl = min(min_odl, np.sqrt(odl))
            # print(min_odl)

            # if np.sqrt(odl) < 1.1:
            #     czy_zatrzymanie = True
            # Aktualizujemy wartosci Q dla aktualnego stanu i wybranej akcji:
            # ........................................................
            # ........................................................
            # w = w + ...
            state_action = np.append(stan, [kat, V])
            features = get_prototype_features(state_action)

            # If next state is terminal, target is just the reward
            if czy_zatrzymanie:
                target = R
            else:
                # Get max Q-value for the next state
                max_q_next = -float("inf")
                for kat_next in np.linspace(-np.pi / 4, np.pi / 4, 7):
                    for V_next in [param_fiz.Vmod, -param_fiz.Vmod, 0]:
                        next_state_action = np.append(nowystan, [kat_next, V_next])
                        next_features = get_prototype_features(next_state_action)
                        q_value_next = np.dot(w, next_features)
                        max_q_next = max(max_q_next, q_value_next)
                target = R + max_q_next

            # Update weights using linear function approximation
            td_error = target - np.dot(w, features)
            w += alfa * td_error * features
            stan = nowystan


        # co jakis czas test z wygenerowaniem historii do pliku:
        if epizod % 100 == 0:
            print(f"\nepizod {epizod} epsilon: {epsylon}")
            park_test(param_fiz, stany_poczatkowe, w, "historia_park.txt")

    # sprawdzenie czy system dobrze uogólnia dla dowolnych stanów początkowych:
    stany_pocz_losowe = pm.random_initial_states(pm.GlobalVar(),20)
    print("Test dla losowych stanów początkowych:")
    park_test(param_fiz, stany_pocz_losowe, w, "historia_park_los.txt")
    
ocena_koncowa_maks = pm.final_score(pm.GlobalVar(), [0,0,-np.pi], if_collision=False, num_of_steps=100)
print("najlepsza możliwa ocena końcowa = " + str(ocena_koncowa_maks))

park_train()



