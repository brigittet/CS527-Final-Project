import pygame
import neat
import os
import sys
import random
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLOK_SIZE = 20
WIDHT = 15
HIEGHT = 15
SCRN_WIDTH = WIDHT * BLOK_SIZE
SCRN_HIEGHT = HIEGHT * BLOK_SIZE

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREN = (0, 255, 0)
DARK_GREN = (0, 180, 0)
RED = (255, 0, 0)
BLU = (100, 150, 255)
GRAY = (40, 40, 40)

gen_count = 0
best_scor_ever = 0
best_genom_ever = None
all_best_fitnes = []
all_avg_fitnes = []

class Snak:
    def __init__(self):
        startX = WIDHT // 2
        startY = HIEGHT // 2
        self.bdy = [
            [startX, startY],
            [startX - 1, startY],
            [startX - 2, startY]
        ]
        self.direcn = 'RIGHT'
        self.is_alive = True
        self.scor = 0
        self.moves_remainng = 200
        self.totl_moves = 0
        
    def getHead(self):
        return self.bdy[0]
    
    def changeDir(self, newDir):
        oposites = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if newDir != oposites.get(self.direcn):
            self.direcn = newDir
    
    def moove(self):
        if not self.is_alive:
            return
        
        hed = self.bdy[0].copy()
        
        if self.direcn == 'UP':
            hed[1] -= 1
        elif self.direcn == 'DOWN':
            hed[1] += 1
        elif self.direcn == 'LEFT':
            hed[0] -= 1
        elif self.direcn == 'RIGHT':
            hed[0] += 1
        
        if hed[0] < 0 or hed[0] >= WIDHT or hed[1] < 0 or hed[1] >= HIEGHT:
            self.is_alive = False
            return
        
        if hed in self.bdy[:-1]:
            self.is_alive = False
            return
        
        self.bdy.insert(0, hed)
        self.bdy.pop()
        
        self.moves_remainng -= 1
        self.totl_moves += 1
        
        if self.moves_remainng <= 0:
            self.is_alive = False
    
    def grw(self):
        self.bdy.append(self.bdy[-1].copy())
        self.scor += 1
        self.moves_remainng = min(self.moves_remainng + 150, 400)
    
    def drw(self, scrn):
        for i, seg in enumerate(self.bdy):
            x = seg[0] * BLOK_SIZE
            y = seg[1] * BLOK_SIZE
            colr = DARK_GREN if i == 0 else GREN
            pygame.draw.rect(scrn, colr, (x + 1, y + 1, BLOK_SIZE - 2, BLOK_SIZE - 2))


class Fod:
    def __init__(self, snake_bdy):
        self.respwn(snake_bdy)
    
    def respwn(self, snake_bdy):
        attemps = 0
        while attemps < 100:
            self.x = random.randint(0, WIDHT - 1)
            self.y = random.randint(0, HIEGHT - 1)
            if [self.x, self.y] not in snake_bdy:
                break
            attemps += 1
    
    def drw(self, scrn):
        x = self.x * BLOK_SIZE
        y = self.y * BLOK_SIZE
        pygame.draw.rect(scrn, RED, (x + 1, y + 1, BLOK_SIZE - 2, BLOK_SIZE - 2))


def getInputs(snak, fod):
    hed = snak.getHead()
    
    def isDangr(x, y):
        if x < 0 or x >= WIDHT or y < 0 or y >= HIEGHT:
            return 1.0
        if [x, y] in snak.bdy[:-1]:
            return 1.0
        return 0.0
    
    dangr_up = isDangr(hed[0], hed[1] - 1)
    dangr_down = isDangr(hed[0], hed[1] + 1)
    dangr_left = isDangr(hed[0] - 1, hed[1])
    dangr_right = isDangr(hed[0] + 1, hed[1])
    
    fod_dir_x = 0
    fod_dir_y = 0
    if fod.x < hed[0]:
        fod_dir_x = -1
    elif fod.x > hed[0]:
        fod_dir_x = 1
    if fod.y < hed[1]:
        fod_dir_y = -1
    elif fod.y > hed[1]:
        fod_dir_y = 1
    
    dir_x = 0
    dir_y = 0
    if snak.direcn == 'LEFT':
        dir_x = -1
    elif snak.direcn == 'RIGHT':
        dir_x = 1
    elif snak.direcn == 'UP':
        dir_y = -1
    elif snak.direcn == 'DOWN':
        dir_y = 1
    
    return [
        dangr_up, dangr_down, dangr_left, dangr_right,
        fod_dir_x, fod_dir_y,
        dir_x, dir_y
    ]

def getDirFromOutput(outpt):
    directons = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    return directons[outpt.index(max(outpt))]

def calcDist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def evalGenoms(genoms, config):
    global gen_count, best_scor_ever, best_genom_ever
    global all_best_fitnes, all_avg_fitnes
    gen_count += 1
    
    fitnes_list = []
    best_scor_this_gen = 0
    
    for genom_id, genom in genoms:
        net = neat.nn.FeedForwardNetwork.create(genom, config)
        
        totl_scor = 0
        totl_fitnes = 0
        num_gams = 3
        
        for gam in range(num_gams):
            snak = Snak()
            fod = Fod(snak.bdy)
            
            fitnes = 0
            prev_dist = calcDist(snak.getHead(), [fod.x, fod.y])
            
            while snak.is_alive:
                inpts = getInputs(snak, fod)
                outpt = net.activate(inpts)
                new_dir = getDirFromOutput(list(outpt))
                snak.changeDir(new_dir)
                
                snak.moove()
                
                if not snak.is_alive:
                    break
                
                hed = snak.getHead()
                if hed[0] == fod.x and hed[1] == fod.y:
                    snak.grw()
                    fod.respwn(snak.bdy)
                    fitnes += 100 + (snak.scor * 10)
                    prev_dist = calcDist(hed, [fod.x, fod.y])
                else:
                    curr_dist = calcDist(hed, [fod.x, fod.y])
                    if curr_dist < prev_dist:
                        fitnes += 1
                    else:
                        fitnes -= 1.2
                    prev_dist = curr_dist
            
            totl_scor += snak.scor
            totl_fitnes += fitnes
        
        avg_scor = totl_scor / num_gams
        avg_fitnes = totl_fitnes / num_gams
        
        if totl_scor > best_scor_this_gen:
            best_scor_this_gen = totl_scor
        
        if avg_scor >= best_scor_ever / num_gams:
            test_scors = []
            for _ in range(5):
                snak = Snak()
                fod = Fod(snak.bdy)
                while snak.is_alive:
                    inpts = getInputs(snak, fod)
                    outpt = net.activate(inpts)
                    new_dir = getDirFromOutput(list(outpt))
                    snak.changeDir(new_dir)
                    snak.moove()
                    if snak.is_alive:
                        hed = snak.getHead()
                        if hed[0] == fod.x and hed[1] == fod.y:
                            snak.grw()
                            fod.respwn(snak.bdy)
                test_scors.append(snak.scor)
            
            max_tst = max(test_scors)
            if max_tst > best_scor_ever:
                best_scor_ever = max_tst
                best_genom_ever = genom
                with open('best_snake.pkl', 'wb') as f:
                    pickle.dump(genom, f)
                print(f"  *** NEW BEST! Score: {max_tst} (tested 5 games: {test_scors}) ***")
        
        genom.fitness = max(0, avg_fitnes)
        fitnes_list.append(genom.fitness)
    
    all_best_fitnes.append(max(fitnes_list))
    all_avg_fitnes.append(sum(fitnes_list) / len(fitnes_list))
    
    print(f"Gen {gen_count} | Best This Gen: {best_scor_this_gen} | Best Ever: {best_scor_ever} | Avg Fitness: {sum(fitnes_list)/len(fitnes_list):.0f}")


def plotFitnes():
    if not all_best_fitnes:
        return
    
    plt.figure(figsize=(10, 6))
    gens = range(1, len(all_best_fitnes) + 1)
    plt.plot(gens, all_best_fitnes, 'r-', linewidth=2, label='Best Fitness')
    plt.plot(gens, all_avg_fitnes, 'b-', linewidth=2, label='Average Fitness')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('NEAT Snake - Fitness Over Generations', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('snake_fitness_plot.png', dpi=150)
    print("Saved: snake_fitness_plot.png")
    plt.close()


def runNeat(config_pth):
    global best_scor_ever, best_genom_ever
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_pth
    )
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    print("NEAT SNAKE - EXTENDED TRAINING (100 generations)")
    print("Population: 150 snakes")
    print("Generations: 100")
    print("Each snake plays 3 games (averaged)")
    print("Best snake tested 5 times before saving")
 
    
    try:
        winnr = pop.run(evalGenoms, 100)
    except KeyboardInterrupt:
        print("\nStopped early. Saving best snake.")
    except Exception as e:
        print(f"\nError: {e}")
    
    if best_genom_ever:
        with open('best_snake.pkl', 'wb') as f:
            pickle.dump(best_genom_ever, f)
        print(f"\nFINAL: Best snake saved! Score: {best_scor_ever}")
    
    plotFitnes()


def replayBest(config_pth):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_pth
    )
    
    try:
        with open('best_snake.pkl', 'rb') as f:
            winnr = pickle.load(f)
    except FileNotFoundError:
        print("No saved snake! Run training first.")
        return
    
    net = neat.nn.FeedForwardNetwork.create(winnr, config)
    
    pygame.init()
    scrn = pygame.display.set_mode((SCRN_WIDTH, SCRN_HIEGHT + 50))
    pygame.display.set_caption("NEAT Snake - Best AI")
    clk = pygame.time.Clock()
    fnt = pygame.font.SysFont('Arial', 20)
    
    snak = Snak()
    fod = Fod(snak.bdy)
    spd = 10
    
    print("\nWatching BEST snake play!")
    print("Press UP/DOWN to change speed\n")
    
    while snak.is_alive:
        for evnt in pygame.event.get():
            if evnt.type == pygame.QUIT:
                pygame.quit()
                return
            if evnt.type == pygame.KEYDOWN:
                if evnt.key == pygame.K_UP:
                    spd = min(30, spd + 2)
                elif evnt.key == pygame.K_DOWN:
                    spd = max(3, spd - 2)
        
        inpts = getInputs(snak, fod)
        outpt = net.activate(inpts)
        new_dir = getDirFromOutput(list(outpt))
        snak.changeDir(new_dir)
        
        snak.moove()
        
        if snak.is_alive:
            hed = snak.getHead()
            if hed[0] == fod.x and hed[1] == fod.y:
                snak.grw()
                fod.respwn(snak.bdy)
                print(f"Ate food! Score: {snak.scor}, Length: {len(snak.bdy)}")
        
        scrn.fill(BLACK)
        for x in range(0, SCRN_WIDTH, BLOK_SIZE):
            pygame.draw.line(scrn, GRAY, (x, 0), (x, SCRN_HIEGHT))
        for y in range(0, SCRN_HIEGHT, BLOK_SIZE):
            pygame.draw.line(scrn, GRAY, (0, y), (SCRN_WIDTH, y))
        
        fod.drw(scrn)
        snak.drw(scrn)
        
        txt = fnt.render(f"Score: {snak.scor}   Length: {len(snak.bdy)}   Speed: {spd}", True, WHITE)
        scrn.blit(txt, (10, SCRN_HIEGHT + 12))
        
        pygame.display.flip()
        clk.tick(spd)
    
    print(f"\nGame Over! Final Score: {snak.scor}, Length: {len(snak.bdy)}")
    
    scrn.fill(BLACK)
    txt = fnt.render(f"GAME OVER! Score: {snak.scor}  Length: {len(snak.bdy)}", True, RED)
    scrn.blit(txt, (SCRN_WIDTH // 2 - 140, SCRN_HIEGHT // 2))
    pygame.display.flip()
    pygame.time.delay(3000)
    pygame.quit()


if __name__ == "__main__":
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_pth = os.path.join(local_dir, "config-feedforward.txt")
    
   
    print(" NEAT SNAKE")
  
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replayBest(config_pth)
    else:
        runNeat(config_pth)