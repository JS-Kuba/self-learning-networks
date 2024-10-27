# Skrypt do uczenia modelu sterowania układem odwróconego wahadła na wózku

import numpy as np
import pdb

def wah_glob():
    Fmax = 1200
    krokcalk = 0.05
    g = 9.8135
    tar = 0.03
    masawoz = 25
    masawah = 20
    drw = 23
    return Fmax, krokcalk, g, tar, masawoz, masawah, drw


# Obliczenie stanu wahadla w kolejnym kroku czasowym metoda analityczna
# stan - wektor parametrow stanu w czasie t
# stann -   -||- w czasie t + dt
# F - sila dzialajaca na wozek
def wahadlo_model_fizyczny(stan,F):
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()

    if F>Fmax:
        F=Fmax
    if F<-Fmax:
        F=-Fmax

    hh = krokcalk * 0.5
    momwoz = masawoz * drw
    momwah = masawah * drw
    cwoz = masawoz * g
    cwah = masawah * g

    sx=np.sin(stan[0])
    cx=np.cos(stan[0])
    c1=masawoz+masawah*sx*sx
    c2=momwah*stan[1]*stan[1]*sx
    c3=tar*stan[3]*cx

    stany_poczatkoweoch = np.zeros(stan.size)

    stany_poczatkoweoch[0]=stan[1]
    stany_poczatkoweoch[1]=((cwah+cwoz)*sx-c2*cx+c3-F*cx)/(drw*c1)
    stany_poczatkoweoch[2]=stan[3]
    stany_poczatkoweoch[3]=(c2-cwah*sx*cx-c3+F)/c1
    stanh = np.zeros(stan.size)
    for i in range(4):
        stanh[i]=stan[i]+stany_poczatkoweoch[i]*hh
  
    sx=np.sin(stanh[0])
    cx=np.cos(stanh[0])
    c1=masawoz+masawah*sx*sx
    c2=momwah*stanh[1]*stanh[1]*sx
    c3=tar*stanh[3]*cx

    stany_poczatkoweochh = np.zeros(stan.size)
    stany_poczatkoweochh[0]=stanh[1]
    stany_poczatkoweochh[1]=((cwah+cwoz)*sx-c2*cx+c3-F*cx)/(drw*c1)
    stany_poczatkoweochh[2]=stanh[3]
    stany_poczatkoweochh[3]=(c2-cwah*sx*cx-c3+F)/c1
    stann = np.zeros(stan.size)
    for i in range(4):
        stann[i]=stan[i]+stany_poczatkoweochh[i]*krokcalk
    if stann[0] > np.pi:
        stann[0]=stann[0]-2*np.pi
    if stann[0] < -np.pi:
        stann[0]=stann[0]+2*np.pi

    return stann

# nagroda przypisana do stanu - można ją swobodnie kształtować by przyspieszyć uczenie:
def nagroda(stan,nowystan,F):
    # przykładowe wartości nagród/kar pośrednich - można je zmieniać by ułatwić proces uczenia
    # np. uzależniając od fazy uczenia (curriculum learning):
    kara_za_odchylenie = nowystan[0]**2 +  0.25*nowystan[1]**2 + 0.0025* nowystan[2]**2 + 0.0025* nowystan[3]**2
    kara_za_przewrocenie = (abs(nowystan[0]) >= np.pi / 2) * 1000
    # ..............................................
    # ..............................................
    # ..............................................
    return -(kara_za_odchylenie + kara_za_przewrocenie)


def sila(stan, wagi):
    # wyznaczenie siły za pomocą wyuczonego modelu:
    # tutaj należy umieścić model np. model liniowy zwracający siłę dla podanego stanu:
    # .....................................................................
    # ..................................................................... 

    F = 0   # na razie   

    return F

# Test modelu sterowania układem odwróconego wahadła dla podanych stanów początkowcyh
# wynik testu w skali 0-10 zależy od położenia układu w stanie końcowym.
# Wynik jest bliski 10 jeśli wahadło się nie przewróciło i układ znajduje się w stanie zerowym.
# W przypadku przewrócenia się wahadła nagroda końcowa zależy od liczby kroków do przewrócenia
# Historia sterowania zapisywana jest do pliku. Można ją następnie obejrzeć za pomocą programu 
# WizualizacjaWahadla.exe  
def wahadlo_test(stany_poczatkowe, wagi, nazwa_pliku_historii = "historia.txt"):
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()
    pli = open(nazwa_pliku_historii, 'w')
    pli.write("Fmax = " + str(Fmax) + "\n")
    pli.write("krokcalk = " + str(krokcalk) + "\n")
    pli.write("g = " + str(g) + "\n")
    pli.write("tar = " + str(tar) + "\n")
    pli.write("masawoz = " + str(masawoz) + "\n")
    pli.write("masawah = " + str(masawah) + "\n")
    pli.write("drw = " + str(drw) + "\n")

    srednia_nagroda_koncowa = 0
    srednia_liczba_krokow = 0
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape
    for epizod in range(liczba_stanow_poczatkowych):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod
        stan = stany_poczatkowe[nr_stanup, :]

        krok = 0
        czy_przewrocenie_wahadla = 0
        while (krok < 1000) & (czy_przewrocenie_wahadla == False):
            krok = krok + 1

            # Wyznaczamy akcje a (sile) w stanie stan zgodnie z wyuczona strategia 
            # korzystając z aproksymatora aproks (bez eksploracji)
            F = sila(stan, wagi)

            # wyznaczenie nowego stanu:
            nowystan = wahadlo_model_fizyczny(stan, F)
            czy_przewrocenie_wahadla = (abs(nowystan[0]) >= np.pi / 2)
            pli.write(str(epizod + 1) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(stan[3]) + "  " + str(F) + "\n")

            stan = nowystan

        odchylenie_wazone = 0.1*(np.abs(nowystan[0]) +  np.abs(nowystan[1]) + np.abs(nowystan[2]) + np.abs(nowystan[3]))
        kara_za_przewrocenie = czy_przewrocenie_wahadla * (1000 - krok)/10
        nagroda_koncowa = 10/(1 + odchylenie_wazone + kara_za_przewrocenie)

        srednia_nagroda_koncowa += nagroda_koncowa / liczba_stanow_poczatkowych
        srednia_liczba_krokow = srednia_liczba_krokow + krok/liczba_stanow_poczatkowych
        print("w %d epizodzie końcowa nagroda = %g, liczba krokow = %d" %(epizod, nagroda_koncowa, krok))

    print("srednia nagroda końcowa w epizodzie = %g" % (srednia_nagroda_koncowa))
    print("srednia liczba krokow ustania wahadla = %g" % (srednia_liczba_krokow))
    pli.close()

    return srednia_nagroda_koncowa

def wahadlo_uczenie():
    liczba_epizodow = 2000
    alfa = 0.001            # wsp.szybkosci uczenia(moze byc funkcja czasu)
    epsylon = 0.1           # wsp.eksploracji(moze byc funkcja czasu)

    # tablica stanów początkowych - podczas uczenia, zwłaszcza na początku, dobrze by była ustalona - 
    # póżniej można losować stany początkowe, by uzyskać lepsze uogólnianie
    stany_poczatkowe = np.array([[np.pi/6,0, 0, 0],[0, np.pi/3, 0, 0], [0, 0, -10, 1], [0, 0, 0, -10], [np.pi/12, np.pi/6, 0, 0],
                      [np.pi/12, -np.pi/6, 0, 0], [-np.pi/12, np.pi/6, 0, 0], [-np.pi/12, -np.pi/6, 0, 0],
                      [np.pi/12, 0, 0, 0], [0, 0, -10, 10]],dtype=float)

    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape

    # inicjacja kodowania, wyznaczenie liczby parametrów (wag):
    # ........................................................
    # ........................................................

    # inicjacja wektora wag aproksymatora:
    liczba_wag = 1000       # na razie, by sie uruchomilo
    wagi = np.zeros(liczba_wag)

    for epizod in range(liczba_epizodow):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod %  liczba_stanow_poczatkowych
        stan = stany_poczatkowe[nr_stanup, :]

        krok = 0
        czy_wahadlo_przewrocilo_sie = 0
        while (krok < 1000) & (czy_wahadlo_przewrocilo_sie == 0):
            krok = krok + 1

            # Wyznaczamy akcje a (sile) w stanie stan z uwzględnieniem
            # eksploracji (np. metoda epsylon-zachlanna lub softmax)
            # ........................................................
            # ........................................................
            F = sila(stan, wagi) # + eksploracja

            # wyznaczenie nowego stanu:
            nowystan = wahadlo_model_fizyczny(stan, F)

            czy_wahadlo_przewrocilo_sie = (abs(nowystan[0]) >= np.pi / 2)
            R = nagroda(stan, nowystan, F)

            # Aktualizujemy wartosci Q dla aktualnego stanu i wybranej akcji:
            # ........................................................
            # ........................................................
            # wagi = wagi + ...

            stan = nowystan

        # co jakis czas test z wygenerowaniem historii do pliku:
        if epizod % 1000 == 0:
            print("liczba epizodów uczenia = "+str(epizod))
            wahadlo_test(stany_poczatkowe, wagi)

    # test uogólniania:
    print("\nTest uogólniania dla losowaych stanów początkowych:\n")
    wart_maks = np.array([np.pi/3, np.pi/3, 10, 10])
    stany_poczatkowe_losowe = (1-2*np.random.rand(10,4))*wart_maks
    wahadlo_test(stany_poczatkowe_losowe,wagi,nazwa_pliku_historii="historia_st_los.txt")

wahadlo_uczenie()



