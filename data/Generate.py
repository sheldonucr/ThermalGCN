import random
Center = [3, 4, 5, 6, 7, 8, 9]
ChipW = 12
CoreW = 3
Power = [1, 3, 5, 7, 9]
Num = 20


def regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, region, info):
    with open('Chiplet_Core' + str(i) + '.flp', 'a') as CoreRec:
        Width = CO_x0 - BL_x0
        Hight = BL_h
        if Width != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str(BL_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
            info.append(1)
        Width = BL_x0 + BL_w - CO_x0 - CO_w
        Hight = BL_h
        if Width != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str((CO_x0 + CO_w) * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
            info.append(1)
        Width = CO_w
        Hight = CO_y0 - BL_y0
        if Hight != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str(CO_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
            info.append(1)
        Width = CO_w
        Hight = BL_y0 + BL_h - CO_y0 - CO_h
        if Hight != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str(CO_x0 * 1e-3) + " " + str((CO_y0 + CO_h) * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
            info.append(1)
        CoreRec.write("Core" + str(region) + " " + str(CO_w * 1e-3) + " " + str(CO_h * 1e-3) + " " + str(CO_x0 * 1e-3) + " " + str(CO_y0 * 1e-3)+"\n")
        info.append(2)

        return count,info



for i in range(400):
    count = 0
    info = []
    CenterX = random.choice(Center)
    CenterY = random.choice(Center)
    #Region 1
    BL_x0 = 0
    BL_y0 = 0
    BL_w = CenterX
    BL_h = CenterY

    CenterV = []
    CenterH = []
    for j in range(BL_x0, BL_x0 + BL_w - CoreW + 1):
        CenterV.append(j)
    for j in range(BL_y0, BL_y0 + BL_h - CoreW + 1):
        CenterH.append(j)
    CO_x0 = random.choice(CenterV)
    CO_y0 = random.choice(CenterH)
    CO_w = CoreW
    CO_h = CoreW

    count,info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 1,info)

    #Region 2
    BL_x0 = CenterX
    BL_y0 = 0
    BL_w = ChipW - CenterX
    BL_h = CenterY

    CenterV = []
    CenterH = []
    for j in range(BL_x0, BL_x0 + BL_w - CoreW + 1):
        CenterV.append(j)
    for j in range(BL_y0, BL_y0 + BL_h - CoreW + 1):
        CenterH.append(j)
    CO_x0 = random.choice(CenterV)
    CO_y0 = random.choice(CenterH)
    CO_w = CoreW
    CO_h = CoreW

    count,info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 2,info)

    #Region 3
    BL_x0 = CenterX
    BL_y0 = CenterY
    BL_w = ChipW - CenterX
    BL_h = ChipW - CenterY

    CenterV = []
    CenterH = []
    for j in range(BL_x0, BL_x0 + BL_w - CoreW + 1):
        CenterV.append(j)
    for j in range(BL_y0, BL_y0 + BL_h - CoreW + 1):
        CenterH.append(j)
    CO_x0 = random.choice(CenterV)
    CO_y0 = random.choice(CenterH)
    CO_w = CoreW
    CO_h = CoreW

    count,info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 3,info)

    #Region 4
    BL_x0 = 0
    BL_y0 = CenterY
    BL_w = CenterX
    BL_h = ChipW - CenterY

    CenterV = []
    CenterH = []
    for j in range(BL_x0, BL_x0 + BL_w - CoreW + 1):
        CenterV.append(j)
    for j in range(BL_y0, BL_y0 + BL_h - CoreW + 1):
        CenterH.append(j)
    CO_x0 = random.choice(CenterV)
    CO_y0 = random.choice(CenterH)
    CO_w = CoreW
    CO_h = CoreW

    count, info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 4, info)
    
    for j in range(Num):
        with open('Chiplet_Core' + str(i) + "_Power" + str(j) + '.ptrace', 'a') as CorePower:
            temp1 = 0
            temp2 = 0
            for k in range(len(info)):
                if info[k] == 1:
                    temp1 = temp1 + 1
                    CorePower.write("TIM" + str(temp1) + " ")
                elif info[k] == 2:
                    temp2 = temp2 + 1
                    CorePower.write("Core" + str(temp2) + " ")
            CorePower.write("\n")
            for k in range(len(info)):
                if info[k] == 1:
                    CorePower.write(str(0) + " ")
                elif info[k] == 2:
                    CorePower.write(str(random.choice(Power)) + " ")
            CorePower.write("\n")
