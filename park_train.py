import numpy as np
import parking_model as pm

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
    ocena_alfa = alfa_zred - 0.5

    # jeśli V==0 nagroda na podstawie odległości
    
    if czy_kolizja:
        wartosc = -1
    elif czy_zatrzymanie:
        wartosc = min(ocena_odl,ocena_alfa)
    else:
        wartosc = 0

    return wartosc

def choose_action(param_fiz, stan, model):
    # tutaj należy wykorzystać wyuczoną strategię w czystej eksploatacji
    # strategia może być np. reprezentowana aproksymatorem funkcji użyteczności
    # ..........................................
    # ..........................................

    kat = -np.pi/8           # jakiś kąt skrętu kół (na razie)
    V = -param_fiz.Vmod      # jakaś prędkość (na razie)
    czy_zatrzymanie = False  # na razie (można przyjąć True np. gdy |V| < próg)
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
    alfa = 0.001  # wsp.szybkosci uczenia(moze byc funkcja czasu)
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
    liczba_wag = 1000     # na razie, by sie uruchomilo
    w = np.zeros(liczba_wag)

    for epizod in range(liczba_epizodow):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod %  liczba_stanow_poczatkowych
        stan = stany_poczatkowe[nr_stanup, :]

        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        while czy_zatrzymanie == False:
            krok = krok + 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan z uwzględnieniem
            # eksploracji (np. metoda epsylon-zachlanna lub softmax lub jeszcze inna)
            # ........................................................
            # ........................................................
            kat = np.pi/8               # na razie
            V = param_fiz.Vmod;         # na razie

            # wyznaczenie nowego stanu:
            nowystan, sr_obrotu, czy_kolizja = pm.model_of_car(param_fiz, stan, kat, V)

            if (czy_kolizja)|(krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            R = nagroda_za_krok(param_fiz, nowystan, czy_kolizja, czy_zatrzymanie)

            # Aktualizujemy wartosci Q dla aktualnego stanu i wybranej akcji:
            # ........................................................
            # ........................................................
            # w = w + ...

            stan = nowystan

        # co jakis czas test z wygenerowaniem historii do pliku:
        if epizod % 1000 == 0:
            print("epizod %d\n" % epizod)
            park_test(param_fiz, stany_poczatkowe, w, "historia_park.txt")

    # sprawdzenie czy system dobrze uogólnia dla dowolnych stanów początkowych:
    stany_pocz_losowe = pm.random_initial_states(pm.GlobalVar(),20)
    print("Test dla losowych stanów początkowych:")
    park_test(param_fiz, stany_pocz_losowe, w, "historia_park_los.txt")
    
ocena_koncowa_maks = pm.final_score(pm.GlobalVar(), [0,0,-np.pi], if_collision=False, num_of_steps=100)
print("najlepsza możliwa ocena końcowa = " + str(ocena_koncowa_maks))

park_train()




