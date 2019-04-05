/*
Source code for the CoinRun environment.

This is built as a shared library and loaded into python with ctypes.
It exposes a C interface similar to that of a VecEnv.

Also includes a mode that creates a window you can interact with using the keyboard.
*/

#include <QtCore/QMutexLocker>
#include <QtCore/QWaitCondition>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QPushButton>
#include <QtGui/QKeyEvent>
#include <QtGui/QPainter>
#include <QtWidgets/QToolButton>
#include <QtCore/QDir>
#include <QtCore/QThread>
#include <QtCore/QProcess>
#include <QtCore/QDateTime>
#include <QtCore/QElapsedTimer>
#include <QtCore/QDirIterator>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <random>
#include <iostream>
#include <memory>
#include <assert.h>
#include <set>

const int NUM_ACTIONS = 7;
const int MAZE_OFFSET = 1;

static
int DISCRETE_ACTIONS[NUM_ACTIONS * 2] = {
    0, 0,
    +1, 0,  // right
    -1, 0,  // left
    0, +1,  // jump
    +1, +1, // right-jump
    -1, +1, // left-jump
    0, -1,  // down  (step down from a crate)
};

enum GameType {
  CoinRunToTheRight_v0     = 1000,
  CoinRunPlatforms_v0      = 1001,
  CoinRunMaze_v0           = 1002,
};

#define VIDEORES 512         // try 64 to record 64x64 video from manual play
#define VIDEORES_STR "512"

const char SPACE = '.';
const char LADDER = '=';
const char LAVA_SURFACE = '^';
const char LAVA_MIDDLE = '|';
const char WALL_SURFACE = 'S';
const char WALL_MIDDLE = 'A';
const char COIN_OBJ1 = '1';
const char COIN_OBJ2 = '2';
const char SPIKE_OBJ = 'P';
const char FLYING_MONSTER = 'F';
const char WALKING_MONSTER = 'M';
const char SAW_MONSTER = 'G';

const int DOWNSAMPLE = 16;
const float MAXSPEED = 0.5;
const float LADDER_MIXRATE = 0.4;
const float LADDER_V = 0.4;
const float MONSTER_SPEED = 0.15;
const float MONSTER_MIXRATE = 0.1;

const int RES_W = 64;
const int RES_H = 64;

const int MAX_COINRUN_DIFFICULTY = 3;
const int MAX_MAZE_DIFFICULTY = 4;

bool USE_LEVEL_SET = false;
int NUM_LEVELS = 0;
int *LEVEL_SEEDS;

bool RANDOM_TILE_COLORS = false;
bool PAINT_VEL_INFO = false;
bool USE_HIGH_DIF = false;
bool USE_DATA_AUGMENTATION = false;
int DEFAULT_GAME_TYPE = CoinRunToTheRight_v0;

static bool shutdown_flag = false;
static std::string monitor_dir;
static int monitor_csv_policy;

const char* test =
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"A.....................F................A"
"A.................F....................A"
"A.............F........................A"
"AAA..................................AAA"
"AA....................................AA"
"AAA..................................AAA"
"AA....................................AA"
"AAA..................................AAA"
"A.......F..............................A"
"A................F.....................A"
"A.........................F............A"
"A......................................A"
"A......................................A"
"A......................................A"
"A......................................A"
"A......................................A"
"A...................G.......G..........A"
"A.................aSSS^^^^^SSSb........A"
"A....................AAAAAAA...........A"
"A......................................A"
"A...................................F..A"
"A......................................A"
"A........1.1.1M1.1.1.........=...1.....A"
"A......aSSSSSSSSSSSSSb....aSb=..aSb....A"
"A............................=.........A"
"A............................=.........A"
"A....  ......................=.........A"
"A... .. .....=...2.2.2.2.2...=.........A"
"A. ..... ....=aSSSSSSSSSSSSSSb.........A"
"A............=.........................A"
"A............=.........................A"
"A..=.........=...............F.........A"
"A..=...................................A"
"A..=.......#&..........................A"
"A..........$%..........................A"
"A.....#$.#%$#...........S^^^^^^S.......A"
"A.....%#.$&%#.....M..M..A||||||A.......A"
"ASSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS^^A"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
;

static std::vector<int> ground_theme_idxs;
static std::vector<int> walking_theme_idxs;
static std::vector<int> flying_theme_idxs;

bool is_crat(char c) {
  return c=='#' || c=='$' || c=='&' || c=='%';
}

bool is_wall(char c, bool crate_counts=false)
{
  bool wall = c=='S' || c=='A' || c=='a' || c=='b' || c=='a';
  if (crate_counts)
    wall |= is_crat(c);
  return wall;
}
bool is_lethal(char c) {
  return c == LAVA_SURFACE || c == LAVA_MIDDLE || c == SPIKE_OBJ;
}
bool is_coin(char c) {
  return c==COIN_OBJ1 || c==COIN_OBJ2;
}

class VectorOfStates;

class RandGen {
public:
  bool is_seeded = false;
  std::mt19937 stdgen;

  int randint(int low, int high) {
    assert(is_seeded);
    uint32_t x = stdgen();
    uint32_t range = high - low;
    return low + (x % range);
  }

  float rand01() {
    assert(is_seeded);
    uint32_t x = stdgen();
    return (double)(x) / ((double)(stdgen.max()) + 1);
  }

  int randint() {
    assert(is_seeded);
    return stdgen();
  }

  void seed(int seed) {
    stdgen.seed(seed);
    is_seeded = true;
  }
};

static RandGen global_rand_gen;

double get_time() {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time); // you need macOS Sierra 10.12 for this
  return time.tv_sec + 0.000000001 * time.tv_nsec;
}

std::string stdprintf(const char *fmt, ...)
{
  char buf[32768];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  buf[32768 - 1] = 0;
  return buf;
}

inline double sqr(double x)  { return x*x; }
inline double sign(double x)  { return x > 0 ? +1 : (x==0 ? 0 : -1); }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline float clip_abs(float x, float y)  { if (x > y) return y; if (x < -y) return -y; return x; }

class Maze;

const int MONSTER_TRAIL = 14;

class Monster {
public:
  float x, y;
  float prev_x[MONSTER_TRAIL], prev_y[MONSTER_TRAIL];
  float vx = 0.01, vy = 0;
  bool is_flying;
  bool is_walking;
  int theme_n = -1;
  void step(const std::shared_ptr<Maze>& maze);
};

class Maze {
public:
  int spawnpos[2];
  int w, h;
  int game_type;
  int* walls;
  int coins;
  bool is_terminated;

  float gravity;
  float max_jump;
  float air_control;
  float max_dy;
  float max_dx;
  float default_zoom;
  float max_speed;
  float mix_rate;

  std::vector<std::shared_ptr<Monster>> monsters;

  Maze(const int _w, const int _h, int _game_type)
  {
    w = _w;
    h = _h;
    game_type = _game_type;
    walls = new int[w*h];
    is_terminated = false;
    coins = 0;
  }

  ~Maze()
  {
    delete[] walls;
  }

  int& get_elem(int x, int y)
  {
    return walls[w*y + x];
  }

  int set_elem(int x, int y, int val)
  {
    return walls[w*y + x] = val;
  }

  void fill_elem(int x, int y, int dx, int dy, char elem)
  {
    for (int j = 0; j < dx; j++) {
      for (int k = 0; k < dy; k++) {
        set_elem(x + j, y + k, elem);
      }
    }
  }

  bool has_vertical_space(float x, float y, bool crate_counts)
  {
    return !(
      is_wall(get_elem(x + .1, y)) || is_wall(get_elem(x + .9, y))
      || (crate_counts && is_crat(get_elem(x + .1, y)))
      || (crate_counts && is_crat(get_elem(x + .9, y)))
      );
  }

  void init_physics() {
    if (game_type == CoinRunMaze_v0) {
      default_zoom = 7.5;
    } else {
      default_zoom = 5.0;
    }
      
    gravity = .2;
    air_control = .15;
    
    max_jump = 1.5;
    max_speed = .5;
    mix_rate = .2;

    max_dy = max_jump * max_jump / (2*gravity);
    max_dx = max_speed * 2 * max_jump / gravity;
  }
};

class RandomMazeGenerator {
public:
  struct Rec {
    int x;
    int y;
  };

  struct Wall {
      int x1;
      int y1;
      int x2;
      int y2;
  };

  std::vector<Rec> rec_stack;
  std::shared_ptr<Maze> maze;
  RandGen rand_gen;

  int dif;
  int maze_dim;
  int danger_type;

  std::set<int> cell_sets[2000];
  int free_cells[2000];
  int num_free_cells;

  std::set<int> lookup(std::set<int> *cell_sets, int x, int y) {
      return cell_sets[maze_dim*y + x];
  }

  bool is_maze_wall(int x, int y) {
      return is_wall(maze->get_elem(x + MAZE_OFFSET, y + MAZE_OFFSET));
  }

  void set_free_cell(int x, int y) {
      maze->set_elem(x + MAZE_OFFSET, y + MAZE_OFFSET, SPACE);
      free_cells[num_free_cells] = maze_dim*y + x;
      num_free_cells += 1;
  }

  int choose_difficulty(int max_difficulty) {
    return USE_HIGH_DIF ? (max_difficulty) : randn(max_difficulty) + 1;
  }

  void generate_coin_maze() {
    dif = choose_difficulty(MAX_MAZE_DIFFICULTY);
    maze_dim = 2 * (randn(3) + 3 * (dif - 1) + 1) + 1;

    maze->fill_elem(0, 0, maze_dim + 2, maze_dim + 2, WALL_MIDDLE);
    maze->fill_elem(MAZE_OFFSET, MAZE_OFFSET, 1, 1, SPACE);

    std::vector<Wall> walls;
    
    num_free_cells = 0;

    std::set<int> s0;
    s0.insert(0);
    cell_sets[0] = s0;

    for (int i = 1; i < maze_dim * maze_dim; i++) {
        std::set<int> s1;
        s1.insert(i);
        cell_sets[i] = s1;
    }

    for (int i = 1; i < maze_dim; i += 2) {
        for (int j = 0; j < maze_dim; j += 2) {
            if (i > 0 && i < maze_dim - 1) {
                walls.push_back(Wall({i-1,j,i+1,j}));
            }
        }
    }

    for (int i = 0; i < maze_dim; i += 2) {
        for (int j = 1; j < maze_dim; j += 2) {
            if (j > 0 && j < maze_dim - 1) {
                walls.push_back(Wall({i,j-1,i,j+1}));
            }
        }
    }

    while (walls.size() > 0) {
        int n = randn(walls.size());
        Wall wall = walls[n];

        s0 = lookup(cell_sets, wall.x1, wall.y1);
        std::set<int> s1 = lookup(cell_sets, wall.x2, wall.y2);

        int x0 = (wall.x1 + wall.x2) / 2;
        int y0 = (wall.y1 + wall.y2) / 2;
        int center = maze_dim*y0 + x0;
        int p1 = maze_dim*wall.y1 + wall.x1;

        bool can_remove = is_maze_wall(x0, y0) && (s0 != s1);

        if (can_remove) {
            set_free_cell(wall.x1, wall.y1);
            set_free_cell(x0, y0);
            set_free_cell(wall.x2, wall.y2);

            s1.insert(s0.begin(), s0.end());
            s1.insert(center);

            for (int child : s1) {
                cell_sets[child] = s1;  
            }
        }

        walls.erase(walls.begin() + n);
    }

    int m = randn(num_free_cells);
    int coin_cell = free_cells[m];
    maze->set_elem(coin_cell % maze_dim + MAZE_OFFSET, coin_cell / maze_dim + MAZE_OFFSET, COIN_OBJ1);

    maze->spawnpos[0] = MAZE_OFFSET;
    maze->spawnpos[1] = MAZE_OFFSET;
    maze->coins = 1;
  }

  void fill_block_top(int x, int y, int dx, int dy, char fill, char top)
  {
    assert(dy > 0);
    maze->fill_elem(x, y, dx, dy - 1, fill);
    maze->fill_elem(x, y + dy - 1, dx, 1, top);
  }

  void fill_ground_block(int x, int y, int dx, int dy)
  {
    fill_block_top(x, y, dx, dy, WALL_MIDDLE, WALL_SURFACE);
  }

  void fill_lava_block(int x, int y, int dx, int dy)
  {
    fill_block_top(x, y, dx, dy, LAVA_MIDDLE, LAVA_SURFACE);
  }

  void initial_floor_and_walls(int game_type)
  {
    maze->fill_elem(0, 0, maze->w, maze->h, SPACE);

    if (game_type != CoinRunMaze_v0) {
      maze->fill_elem(0, 0, maze->w, 1, WALL_SURFACE);
      maze->fill_elem(0, 0, 1, maze->h, WALL_MIDDLE);
      maze->fill_elem(maze->w - 1, 0, 1, maze->h, WALL_MIDDLE);
      maze->fill_elem(0, maze->h - 1, maze->w, 1, WALL_MIDDLE);
    }

    maze->init_physics();
  }

  int randn(int n) {
    return rand_gen.randint(0, n);
  }

  float rand01() {
    return rand_gen.rand01();
  }

  char choose_crate() {
    return "#$&%"[randn(4)];
  }

  bool jump_and_build_platform_somewhere()
  {
    float gravity = .2;
    float max_jump = 1.5;
    float max_speed = 0.5;

    if (rec_stack.empty()) return false;
    int n = int(sqrt(randn(rec_stack.size() * rec_stack.size())));
    assert(n < (int)rec_stack.size());
    Rec r = rec_stack[n];
    float vx = (rand01()*2-1)*0.5*max_speed;
    float vy = (0.8 + 0.2*rand01())*max_jump;

    int top = 1 + int(vy/gravity);
    int ix, iy;
    if (randn(2)==1) {
      int steps = top + (randn(top/2));
      float x = r.x;
      float y = r.y + 1;

      ix = -1;
      iy = -1;
      for (int s=0; s<steps; s++) {
        vy -= gravity;
        x += vx;
        y += vy;
        if (ix != int(x) || iy != int(y)) {
          ix = int(x);
          iy = int(y);
          bool ouch = false;
          ouch |= ix<1;
          ouch |= ix>=maze->w-1;
          ouch |= iy<1;
          ouch |= iy>=maze->h-1;
          if (ouch) return false;
          char c = maze->get_elem(ix, iy);
          ouch |= c!=SPACE && c!=' ';
          if (ouch) return false;
          maze->set_elem(ix, iy, ' ');
        }
      }
    } else {
      ix = r.x;
      iy = r.y;
      if (is_crat(maze->get_elem(ix, iy)) || is_crat(maze->get_elem(ix, iy-1)))
        return false; // don't build ladders starting from crates
      rec_stack.erase(rec_stack.begin()+n);
      std::vector<Rec> future_ladder;
      int ladder_len = 5 + randn(10);
      for (int s=0; s<ladder_len; s++) {
        future_ladder.push_back(Rec({ ix, iy }));
        iy += 1;
        bool ouch = false;
        ouch |= iy>=maze->h-3;
        ouch |= maze->get_elem(ix, iy) != SPACE;
        ouch |= maze->get_elem(ix-1, iy) == LADDER;
        ouch |= maze->get_elem(ix+1, iy) == LADDER;
        if (ouch) return false;
      }
      for (const Rec& f: future_ladder)
        maze->set_elem(f.x, f.y, LADDER);
      maze->set_elem(ix, iy, LADDER);
    }

    char c = maze->get_elem(ix, iy);
    if (c==SPACE || c==' ')
      maze->set_elem(ix, iy, vx>0 ? 'a':'b');
    std::vector<Rec> crates;
    std::vector<Rec> monster_candidates;
    int len = 2 + randn(10);
    int crates_shift = randn(20);
    for (int platform=0; platform<len; platform++) {
      ix += (vx>0 ? +1 : -1);
      int c = maze->get_elem(ix, iy);
      if (c == ' ' || c == SPACE) {
        maze->set_elem(ix, iy, (platform<len-1) ? WALL_SURFACE : (vx>0 ? 'b':'a'));
        rec_stack.push_back(Rec({ ix, iy+1 }));
        if (int(ix*0.2 + iy + crates_shift) % 4 == 0)
          crates.push_back(Rec({ ix, iy+1 }));
        else if (platform>0 && platform<len-1)
          monster_candidates.push_back(Rec({ ix, iy+1 }));
      } else {
        if (c =='a' || c == 'b')
          maze->set_elem(ix, iy, WALL_SURFACE);
        break;
      }
    }

    if (monster_candidates.size() > 1) {
      const Rec& r = monster_candidates[randn(monster_candidates.size())];
      maze->set_elem(r.x, r.y, WALKING_MONSTER);
    }

    while (1) {
      int cnt = crates.size();
      if (cnt==0) break;
      for (int c=0; c<(int)crates.size(); ) {
        char w = maze->get_elem(crates[c].x, crates[c].y);
        char wl = maze->get_elem(crates[c].x-1, crates[c].y);
        char wr = maze->get_elem(crates[c].x+1, crates[c].y);
        char wu = maze->get_elem(crates[c].x, crates[c].y+1);
        int want = 2 + is_crat(wl) + is_crat(wr) - (wr==LADDER) - (wl==LADDER) - is_wall(wu);
        if (randn(4) < want && crates[c].y < maze->h-2) {
          if (w==' ' || w==SPACE)
            maze->set_elem(crates[c].x, crates[c].y, choose_crate());
          crates[c].y += 1;
          rec_stack.push_back(Rec({ crates[c].x, crates[c].y }));  // coins on crates, jumps from crates
          c++;
        } else {
          crates.erase(crates.begin() + c);
        }
      }
    }

    return true;
  }

  void place_coins()
  {
    int coins = 0;
    while (!rec_stack.empty()) {
      Rec r = rec_stack[rec_stack.size()-1];
      rec_stack.pop_back();
      int x = r.x;
      int y = r.y;
      bool good_place =
        maze->get_elem(x, y) == SPACE &&
        r.y > 2 &&
        maze->get_elem(x-1, y) == SPACE &&
        maze->get_elem(x+1, y) == SPACE &&
        maze->get_elem(x, (y+1)) == SPACE &&
        is_wall(maze->get_elem(x-1, y-1), true) &&
        is_wall(maze->get_elem(x, y-1), true) &&
        is_wall(maze->get_elem(x+1, y-1), true);
      if (good_place) {
        maze->set_elem(x, y, '1');
        coins += 1;
      }
    }
    maze->coins = coins;
  }

  void remove_traces_add_monsters()
  {
    maze->monsters.clear();
    for (int y=1; y<maze->h; ++y) {
      for (int x=1; x<maze->w-1; x++) {
        int& c = maze->get_elem(x, y);
        int& b = maze->get_elem(x, y-1);
        int cl = maze->get_elem(x-1, y);
        int cr = maze->get_elem(x+1, y);

        if (c==' ' && randn(20)==0 && !is_wall(b) && y>2) {
          maze->set_elem(x, y, FLYING_MONSTER);
        } else if (c==' ') {
          maze->set_elem(x, y, SPACE);
        }
        if ((c=='a' || c=='b') && is_wall(b))
          c = 'S';
        if (is_wall(c) && is_wall(b))
          b = 'A';
        if (c==FLYING_MONSTER || c==WALKING_MONSTER || c==SAW_MONSTER) {
          std::shared_ptr<Monster> m(new Monster);
          m->x = x;
          m->y = y;
          for (int t=0; t<MONSTER_TRAIL; t++) {
            m->prev_x[t] = x;
            m->prev_y[t] = y;
          }
          m->is_flying = c==FLYING_MONSTER;
          m->is_walking = c==WALKING_MONSTER;

          std::vector<int> *type_theme_idxs;

          if (m->is_flying) {
            type_theme_idxs = &flying_theme_idxs;
          } else if (m->is_walking) {
            type_theme_idxs = &walking_theme_idxs;
          } else {
            type_theme_idxs = &ground_theme_idxs;
          }

          int chosen_idx = randn(type_theme_idxs->size());
          m->theme_n = (*type_theme_idxs)[chosen_idx];

          c = SPACE;

          if (!m->is_walking || (!is_wall(cl) && !is_wall(cr))) // walking monster should have some free space to move
            maze->monsters.push_back(m);
        }
      }
    }
  }

  void generate_test_level()
  {
    maze->spawnpos[0] = 2;
    maze->spawnpos[1] = 2;
    maze->coins = 0;
    for (int y=0; y<maze->h; ++y) {
      for (int x=0; x<maze->w; x++) {
        char c = test[maze->w*(maze->h-y-1) + x];
        if (is_coin(c)) maze->coins += 1;
          maze->set_elem(x, y, c);
      }
    }
    remove_traces_add_monsters();
  }

  void generate_coins_on_platforms()
  {
    maze->spawnpos[0] = 1 + randn(maze->w - 2);
    maze->spawnpos[1] = 1;

    for (int x=0; x<maze->w; x++) {
      rec_stack.push_back(Rec({ x, 1 }));
    }

    int want_platforms = 11;
    for (int p=0; p<want_platforms*10; p++) {
      bool success = jump_and_build_platform_somewhere();
      if (success) want_platforms -= 1;
      if (want_platforms==0) break;
    }

    remove_traces_add_monsters();
    place_coins();
  }

  void generate_coin_to_the_right(int game_type)
  {
    maze->spawnpos[0] = 1;
    maze->spawnpos[1] = 1;
    maze->coins = 1;

    dif = choose_difficulty(MAX_COINRUN_DIFFICULTY);
    int num_sections = randn(dif) + dif;
    int curr_x = 5;
    int curr_y = 1;

    int pit_threshold = dif;
    int danger_type = randn(3);

    char secondary_monster_type = WALKING_MONSTER;
    int max_dy = (maze->max_dy - .5); 
    int max_dx = (maze->max_dx - .5);

    for (int i = 0; i < num_sections; i++) {
      if (curr_x + 15 >= maze->w) {
        break;
      }

      int dy = randn(4) + 1 + int(dif / 3);

      if (dy > max_dy) {
        dy = max_dy;
      }

      if (curr_y >= 20) {
        dy *= -1;
      } else if (curr_y >= 5 and randn(2) == 1) {
        dy *= -1;
      }

      int dx = randn(2 * dif) + 3 + int(dif / 3);

      curr_y += dy;

      if (curr_y < 1) {
        curr_y = 1;
      }

      bool use_pit = (dx > 7) && (curr_y > 3) && (randn(20) >= pit_threshold);

      if (use_pit) {
        int x1 = randn(3) + 1;
        int x2 = randn(3) + 1;
        int pit_width = dx - x1 - x2;

        if (pit_width > max_dx) {
          pit_width = max_dx;
          x2 = dx - x1 - pit_width;
        }

        fill_ground_block(curr_x, 0, x1, curr_y);
        fill_ground_block(curr_x + dx - x2, 0, x2, curr_y);

        int lava_height = randn(curr_y - 3) + 1;

        if (danger_type == 0) {
          fill_lava_block(curr_x + x1, 1, pit_width, lava_height);
        } else if (danger_type == 1) {
          maze->fill_elem(curr_x + x1, 1, pit_width, 1, SAW_MONSTER);
        } else if (danger_type == 2) {
          maze->fill_elem(curr_x + x1, 1, pit_width, 1, secondary_monster_type);
        }

        if (pit_width > 4) {
          int x3, w1;
          if (pit_width == 5) {
            x3 = 1 + randn(2);
            w1 = 1 + randn(2);
          } else if (pit_width == 6) {
            x3 = 2 + randn(2);
            w1 = 1 + randn(2);
          } else {
            x3 = 2 + randn(2);
            int x4 = 2 + randn(2);
            w1 = pit_width - x3 - x4;
          }

          fill_ground_block(curr_x + x1 + x3, curr_y - 1, w1, 1);
        }

      } else {
        fill_ground_block(curr_x, 0, dx, curr_y);

        int ob1_x = -1;
        int ob2_x = -1;

        if (randn(10) < (2 * dif) && dx > 3) {
          ob1_x = curr_x + randn(dx - 2) + 1;
          maze->set_elem(ob1_x, curr_y, SAW_MONSTER);
        }

        if (randn(10) < dif && dx > 3 && (max_dx >= 4)) {
          ob2_x = curr_x + randn(dx - 2) + 1;
          maze->set_elem(ob2_x, curr_y, secondary_monster_type);
        }

        for (int i = 0; i < 2; i++) {
          int crate_x = curr_x + randn(dx - 1) + 1;

          if (randn(2) == 1 && ob1_x != crate_x && ob2_x != crate_x) {
            int pile_height = randn(3) + 1;

            for (int j = 0; j < pile_height; j++) {
              maze->set_elem(crate_x, curr_y + j, choose_crate());
            }
          }
        }
      }

      curr_x += dx;
    }

    maze->set_elem(curr_x, curr_y, COIN_OBJ1);

    fill_ground_block(curr_x, 0, 1, curr_y);
    maze->fill_elem(curr_x + 1, 0, 1, maze->h, WALL_MIDDLE);

    remove_traces_add_monsters();
  }
};

static QString resource_path;

struct PlayerTheme {
  QString theme_name;
  QImage stand;
  QImage front;
  QImage walk1;
  QImage walk2;
  QImage climb1;
  QImage climb2;
  QImage jump;
  QImage duck;
  QImage hit;
};

struct GroundTheme {
  QString theme_name;
  std::map<char, QImage> walls;
  QImage default_wall;
};

struct EnemyTheme {
  QString enemy_name;
  QImage walk1;
  QImage walk2;
  int anim_freq = 1;
};

static std::vector<GroundTheme> ground_themes;
static std::vector<PlayerTheme> player_themesl;
static std::vector<PlayerTheme> player_themesr;
static std::vector<EnemyTheme> enemy_themel;
static std::vector<EnemyTheme> enemy_themer;

static std::vector<GroundTheme> ground_themes_down;
static std::vector<PlayerTheme> player_themesl_down;
static std::vector<PlayerTheme> player_themesr_down;
static std::vector<EnemyTheme> enemy_themel_down;
static std::vector<EnemyTheme> enemy_themer_down;

static std::vector<QImage> bg_images;
static std::vector<QString> bg_images_fn;

static
QImage downsample(QImage img)
{
  int w = img.width();
  assert(w > 0);
  int h = img.height();

  return img.scaled(w / DOWNSAMPLE, h / DOWNSAMPLE, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

static
void ground_theme_downsample(GroundTheme *g, GroundTheme *d)
{
  for (const std::pair<char, QImage> &pair : g->walls) {
    d->walls[pair.first] = downsample(pair.second);
  }
  d->default_wall = downsample(g->default_wall);
}

static
void player_theme_downsample(PlayerTheme *theme, PlayerTheme *theme_down)
{
  theme_down->stand = downsample(theme->stand);
  theme_down->front = downsample(theme->front);
  theme_down->walk1 = downsample(theme->walk1);
  theme_down->walk2 = downsample(theme->walk2);
  theme_down->climb1 = downsample(theme->climb1);
  theme_down->climb2 = downsample(theme->climb2);
  theme_down->jump = downsample(theme->jump);
  theme_down->duck = downsample(theme->duck);
  theme_down->hit = downsample(theme->hit);
}

static
void enemy_theme_downsample(EnemyTheme* theme, EnemyTheme* theme_down)
{
  theme_down->walk1 = downsample(theme->walk1);
  theme_down->walk2 = downsample(theme->walk2);
}

static
PlayerTheme* choose_player_theme(int theme_n, bool is_facing_right, bool lowres)
{
  std::vector<PlayerTheme>* active_theme;

  if (lowres) {
    active_theme = is_facing_right ? &player_themesr_down : &player_themesl_down;
  } else {
    active_theme = is_facing_right ? &player_themesr : &player_themesl;
  }

  return &(*active_theme)[theme_n];
}

static
GroundTheme* choose_ground_theme(int theme_n, bool lowres)
{
  return lowres ? &ground_themes_down[theme_n] : &ground_themes[theme_n];
}

static
EnemyTheme* choose_enemy_theme(const std::shared_ptr<Monster>& m, int monster_n, bool lowres)
{
  if (lowres) {
    return m->vx>0 ? &enemy_themer_down[m->theme_n] : &enemy_themel_down[m->theme_n];
  } else {
    return m->vx>0 ? &enemy_themer[m->theme_n] : &enemy_themel[m->theme_n];
  }
}

QImage load_resource(QString relpath)
{
  auto path = resource_path + "/" + relpath;
  auto img = QImage(path);
  if (img.width() == 0) {
    fprintf(stderr, "failed to load image %s\n", path.toUtf8().constData());
    exit(EXIT_FAILURE);
  }
  return img;
}

void load_enemy_themes(const char **ethemes, std::vector<int> &type_theme_idxs, bool is_flying_type, bool is_walking_type) {
  for (const char **theme=ethemes; *theme; ++theme) {
    int curr_idx = enemy_themel.size();
    type_theme_idxs.push_back(curr_idx);

    QString dir = "kenney/Enemies/";
    EnemyTheme e1;
    e1.enemy_name = QString::fromUtf8(*theme);
    if (is_walking_type)
      e1.anim_freq = 5;
    e1.walk1 = load_resource(dir + e1.enemy_name + ".png");
    e1.walk2 = load_resource(dir + e1.enemy_name + "_move.png");
    enemy_themel.push_back(e1);
    EnemyTheme e1d = e1;

    EnemyTheme e2 = e1;
    e2.walk1 = e2.walk1.mirrored(true, false);
    e2.walk2 = e2.walk2.mirrored(true, false);
    EnemyTheme e2d = e2;
    enemy_themer.push_back(e2);

    enemy_theme_downsample(&e1, &e1d);
    enemy_theme_downsample(&e2, &e2d);
    enemy_themel_down.push_back(e1d);
    enemy_themer_down.push_back(e2d);
  }
}

void images_load()
{
  static const char *bgthemes[] = {
      "kenney/Backgrounds/blue_desert.png",
      "kenney/Backgrounds/blue_grass.png",
      "kenney/Backgrounds/blue_land.png",
      "kenney/Backgrounds/blue_shroom.png",
      "kenney/Backgrounds/colored_desert.png",
      "kenney/Backgrounds/colored_grass.png",
      "kenney/Backgrounds/colored_land.png",
      "backgrounds/game-backgrounds/seabed.png",
      "backgrounds/game-backgrounds/G049_OT000_002A__background.png",
      "backgrounds/game-backgrounds/Background.png",
      "backgrounds/game-backgrounds/Background (4).png",
      "backgrounds/game-backgrounds/BG_only.png",
      "backgrounds/game-backgrounds/bg1.png",
      "backgrounds/game-backgrounds/G154_OT000_002A__background.png",
      "backgrounds/game-backgrounds/Background (5).png",
      "backgrounds/game-backgrounds/Background (2).png",
      "backgrounds/game-backgrounds/Background (3).png",
      "backgrounds/background-from-glitch-assets/background.png",
      "backgrounds/spacebackgrounds-0/deep_space_01.png",
      "backgrounds/spacebackgrounds-0/spacegen_01.png",
      "backgrounds/spacebackgrounds-0/milky_way_01.png",
      "backgrounds/spacebackgrounds-0/ez_space_lite_01.png",
      "backgrounds/spacebackgrounds-0/meyespace_v1_01.png",
      "backgrounds/spacebackgrounds-0/eye_nebula_01.png",
      "backgrounds/spacebackgrounds-0/deep_sky_01.png",
      "backgrounds/spacebackgrounds-0/space_nebula_01.png",
      "backgrounds/space-backgrounds-3/Background-1.png",
      "backgrounds/space-backgrounds-3/Background-2.png",
      "backgrounds/space-backgrounds-3/Background-3.png",
      "backgrounds/space-backgrounds-3/Background-4.png",
      "backgrounds/background-2/airadventurelevel1.png",
      "backgrounds/background-2/airadventurelevel2.png",
      "backgrounds/background-2/airadventurelevel3.png",
      "backgrounds/background-2/airadventurelevel4.png",
      0};
  for (const char **theme=bgthemes; *theme; ++theme) {
    QString path = QString::fromUtf8(*theme);
    bg_images.push_back(load_resource(path));
    bg_images_fn.push_back(path);
  }

  static const char *gthemes[] = {
    "Dirt",
    "Grass",
    "Planet",
    "Sand",
    "Snow",
    "Stone",
    0};
  for (const char **theme=gthemes; *theme; ++theme) {
    GroundTheme t;
    GroundTheme td;
    t.theme_name = QString::fromUtf8(*theme);
    QString walls = "kenney/Ground/" + t.theme_name + "/" + t.theme_name.toLower();
    t.default_wall = load_resource(walls + "Center.png"); // "Ground/Dirt/dirt.png"
    t.walls['a'] = load_resource(walls + "Cliff_left.png");
    t.walls['b'] = load_resource(walls + "Cliff_right.png");
    t.walls[WALL_SURFACE] = load_resource(walls + "Mid.png");
    t.walls['^'] = load_resource(walls + "Half_mid.png");
    QString items = "kenney/Items/";
    t.walls[' '] = load_resource(items + "star.png");
    t.walls[COIN_OBJ1] = load_resource(items + "coinGold.png");
    t.walls[COIN_OBJ2] = load_resource(items + "gemRed.png");
    QString tiles = "kenney/Tiles/";
    t.walls['#'] = load_resource(tiles + "boxCrate.png");
    t.walls['$'] = load_resource(tiles + "boxCrate_double.png");
    t.walls['&'] = load_resource(tiles + "boxCrate_single.png");
    t.walls['%'] = load_resource(tiles + "boxCrate_warning.png");
    t.walls[LAVA_MIDDLE] = load_resource(tiles + "lava.png");
    t.walls[LAVA_SURFACE] = load_resource(tiles + "lavaTop_low.png");
    t.walls[SPIKE_OBJ] = load_resource(tiles + "spikes.png");
    t.walls[LADDER] = load_resource(tiles + "ladderMid.png");
    ground_themes.push_back(t);
    ground_theme_downsample(&t, &td);
    ground_themes_down.push_back(td);
  }

  static const char *pthemes[] = {
      "Beige",
      "Blue",
      "Green",
      "Pink",
      "Yellow",
      0};
  for (const char **theme=pthemes; *theme; ++theme) {
    PlayerTheme t1;
    PlayerTheme t1d;
    t1.theme_name = QString::fromUtf8(*theme);
    QString dir = "kenney/Players/128x256/" + t1.theme_name + "/alien" + t1.theme_name;
    t1.stand = load_resource(dir + "_stand.png");
    t1.front = load_resource(dir + "_front.png");
    t1.walk1 = load_resource(dir + "_walk1.png");
    t1.walk2 = load_resource(dir + "_walk2.png");
    t1.climb1 = load_resource(dir + "_climb1.png");
    t1.climb2 = load_resource(dir + "_climb2.png");
    t1.jump = load_resource(dir + "_jump.png");
    t1.duck = load_resource(dir + "_duck.png");
    t1.hit = load_resource(dir + "_hit.png");
    player_themesr.push_back(t1);

    PlayerTheme t2;
    PlayerTheme t2d;
    t2.theme_name = QString::fromUtf8(*theme);
    t2.stand = t1.stand.mirrored(true, false);
    t2.front = t1.front.mirrored(true, false);
    t2.walk1 = t1.walk1.mirrored(true, false);
    t2.walk2 = t1.walk2.mirrored(true, false);
    t2.climb1 = t1.climb1.mirrored(true, false);
    t2.climb2 = t1.climb2.mirrored(true, false);
    t2.jump = t1.jump.mirrored(true, false);
    t2.duck = t1.duck.mirrored(true, false);
    t2.hit = t1.hit.mirrored(true, false);
    player_themesl.push_back(t2);

    player_theme_downsample(&t1, &t1d);
    player_themesr_down.push_back(t1d);
    player_theme_downsample(&t2, &t2d);
    player_themesl_down.push_back(t2d);
  }

  static const char *ground_monsters[] = {
    "sawHalf",
    0};

  static const char *flying_monsters[] = {
    "bee",
    0};

  static const char *walking_monsters[] = {
    "slimeBlock",
    "slimePurple",
    "slimeBlue",
    "slimeGreen",
    "mouse",
    "snail",
    "ladybug",
    "wormGreen",
    "wormPink",
    0};

  load_enemy_themes(ground_monsters, ground_theme_idxs, false, false);
  load_enemy_themes(walking_monsters, walking_theme_idxs, false, true);
  load_enemy_themes(flying_monsters, flying_theme_idxs, true, false);
}

struct Agent {
  std::shared_ptr<Maze> maze;
  int theme_n;
  float x, y, vx, vy;
  float spring = 0;
  float zoom = 1.0;
  float target_zoom = 1.0;
  uint8_t render_buf[RES_W*RES_H*4];
  uint8_t* render_hires_buf = 0;
  bool game_over = false;
  float reward = 0;
  float reward_sum = 0;
  bool is_facing_right;
  bool ladder_mode;
  int action_dx = 0;
  int action_dy = 0;
  int time_alive;
  bool support;
  FILE *monitor_csv = 0;
  double t0;

  ~Agent() {
    if (render_hires_buf) {
      delete[] render_hires_buf;
      render_hires_buf = 0;
    }
    if (monitor_csv) {
      fclose(monitor_csv);
      monitor_csv = 0;
    }
  }

  void monitor_csv_open(int n_in_vec) {
    t0 = get_time();
    std::string monitor_fn;
    char *rank_ch = getenv("PMI_RANK");
    if (rank_ch) {
      int rank = atoi(rank_ch);
      monitor_fn = monitor_dir + stdprintf("/%02i%02i.monitor.csv", rank, n_in_vec);
    } else {
      monitor_fn = monitor_dir + stdprintf("/%03i.monitor.csv", n_in_vec);
    }
    monitor_csv = fopen(monitor_fn.c_str(), "wt");
    fprintf(monitor_csv, "# {\"t_start\": %0.2lf, \"gym_version\": \"coinrun\", \"env_id\": \"coinrun\"}\n", t0);
    fprintf(monitor_csv, "r,l,t\n");
    fflush(monitor_csv);
  }

  void monitor_csv_episode_over() {
    if (!monitor_csv)
      return;
    fprintf(monitor_csv, "%0.1f,%i,%0.1f\n", reward_sum, time_alive, get_time() - t0);
    fflush(monitor_csv);
  }

  void reset(int spawn_n) {
    x = maze->spawnpos[0];
    y = maze->spawnpos[1];
    action_dx = 0;
    action_dy = 0;
    time_alive = 0;
    reward_sum = 0;
    vx = vy = spring = 0;
    is_facing_right = true;
  }

  void eat_coin(int x, int y)
  {
    int obj = maze->get_elem(x, y);

    if (is_lethal(obj)) {
      maze->is_terminated = true;
    }

    if (is_coin(obj)) {
      maze->set_elem(x, y, SPACE);
      maze->coins -= 1;

      if (maze->coins == 0) {
        reward += 10.0f;
        reward_sum += 10.0f;
        maze->is_terminated = true;
      } else {
        reward += 1.0f;
        reward_sum += 1.0f;
      }
    }
  }

  void sub_step(float _vx, float _vy)
  {
    float ny = y + _vy;
    float nx = x + _vx;

    if (_vy < 0 && !maze->has_vertical_space(x, ny, false)) {
      y = int(ny) + 1;
      support = true;
      vy = 0;

    } else if (_vy < 0 && !maze->has_vertical_space(x, ny, true)) {
      if (action_dy >= 0 && int(ny)!=int(y)) {
        y = int(ny) + 1;
        vy = 0;
        support = true;
      } else {  // action_dy < 0, come down from a crate
        support = false;
        y = ny;
      }

    } else if (_vy > 0 && !maze->has_vertical_space(x, ny + 1, false)) {
      y = int(ny);
      while (!maze->has_vertical_space(x, y, false)) {
        y -= 1;
      }
      vy = 0;

    } else {
      y = ny;
    }

    int ix = int(x);
    int iy = int(y);
    int inx = int(nx);

    if (_vx < 0 && is_wall(maze->get_elem(inx, iy)) ) {
      vx = 0;
      x = int(inx) + 1;
    } else if (_vx > 0 && is_wall(maze->get_elem(inx + 1, iy))) {
      vx = 0;
      x = int(inx);
    } else {
      x = nx;
    }

    eat_coin(ix, iy);
    eat_coin(ix, iy+1);
    eat_coin(ix+1, iy);
    eat_coin(ix+1, iy+1);
  }

  void step_maze(int game_type) {
    int nx = x + action_dx;
    int ny = y + action_dy;

    char obj = maze->get_elem(nx, ny);

    if (!is_wall(obj)) {
        x = nx;
        y = ny;
    }

    eat_coin(x, y);
  }

  void step_coinrun(int game_type)
  {
    support = false;

    int near_x = int(x + .5);
    char test_is_ladder1 = maze->get_elem(near_x, int(y + 0.2));
    char test_is_ladder2 = maze->get_elem(near_x, int(y - 0.2));

    if (test_is_ladder1 == LADDER || test_is_ladder2 == LADDER) {
      if (action_dy != 0)
        ladder_mode = true;
    } else {
      ladder_mode = false;
    }

    float max_jump = maze->max_jump;
    float max_speed = maze->max_speed;
    float mix_rate = maze->mix_rate;

    if (ladder_mode) {
      vx = (1-LADDER_MIXRATE)*vx + LADDER_MIXRATE*max_speed*(action_dx + 0.2*(near_x - x));
      vx = clip_abs(vx, LADDER_V);
      vy = (1-LADDER_MIXRATE)*vy + LADDER_MIXRATE*max_speed*action_dy;
      vy = clip_abs(vy, LADDER_V);

    } else if (spring > 0 && vy==0 && action_dy==0) {
      vy = max_jump;

      spring = 0;
      support = true;
    } else {
      vy -= maze->gravity;
    }

    vy = clip_abs(vy, max_jump);
    vx = clip_abs(vx, max_speed);

    int num_sub_steps = 2;
    float pct = 1.0 / num_sub_steps;

    for (int s = 0; s < num_sub_steps; s++) {
      sub_step(vx * pct, vy * pct);
      if (vx == 0 && vy == 0) {
        break;
      }
    }

    if (support) {
      if (action_dy > 0)
        spring += sign(action_dy) * max_jump/4; // four jump heights
      if (action_dy < 0)
        spring = -0.01;
      if (action_dy == 0 && spring < 0)
        spring = 0;

      spring = clip_abs(spring, max_jump);
      vx = (1-mix_rate)*vx;
      if (spring==0) vx += mix_rate*max_speed*action_dx;
      if (fabs(vx) < mix_rate*max_speed) vx = 0;

    } else {
      spring = 0;
      float ac = maze->air_control;
      vx = (1-ac*mix_rate)*vx + ac*mix_rate*action_dx;
    }

    if (vx < 0) {
      is_facing_right = false;
    } else if (vx > 0) {
      is_facing_right = true;
    }
  }

  void step(int game_type)
  {
    time_alive += 1;
    int timeout = 0;

    if (game_type == CoinRunMaze_v0) {
      timeout = 500;
      step_maze(game_type);
    } else {
      timeout = 1000;
      step_coinrun(game_type);
    }

    if (time_alive > timeout) {
        maze->is_terminated = true;
    }
  }

  QImage picture(PlayerTheme *theme) const
  {
    if (ladder_mode)
      return (time_alive / 5 % 2 == 0) ? theme->climb1 : theme->climb2;
    if (vy != 0)
      return theme->jump;
    if (spring != 0)
      return theme->duck;
    if (vx == 0)
      return theme->stand;

    return (time_alive / 5 % 2 == 0) ? theme->walk1 : theme->walk2;
  }
};

void Monster::step(const std::shared_ptr<Maze>& maze)
{
  if (!is_flying && !is_walking)
  return;
  float control = sign(vx);
  int ix = int(x);
  int iy = int(y);
  int look_left  = maze->get_elem(ix-0, iy);
  int look_right = maze->get_elem(ix+1, iy);
  if (is_wall(look_left)) control = +1;
  if (is_wall(look_right)) control = -1;
  if (is_walking) {
    int feel_left  = maze->get_elem(ix-0, iy-1);
    int feel_right = maze->get_elem(ix+1, iy-1);
    if (!is_wall(feel_left)) control = +1;
    if (!is_wall(feel_right)) control = -1;
  }
  vx = clip_abs(MONSTER_MIXRATE*control + (1-MONSTER_MIXRATE)*vx, MONSTER_SPEED);
  x += vx;
  y += vy;
  for (int t=1; t<MONSTER_TRAIL; t++) {
    prev_x[t-1] = prev_x[t];
    prev_y[t-1] = prev_y[t];
  }
  prev_x[MONSTER_TRAIL-1] = x;
  prev_y[MONSTER_TRAIL-1] = y;
}

struct State {
  int state_n; // in vstate
  std::shared_ptr<Maze> maze;
  int ground_n;
  int bg_n;
  State(const std::shared_ptr<VectorOfStates>& belongs_to):
    belongs_to(belongs_to)  { }

  QMutex state_mutex;
   std::weak_ptr<VectorOfStates> belongs_to;
   int time;
   Agent agent;

  QMutex step_mutex;
   bool step_in_progress = false;
   bool agent_ready = false;
};

void state_reset(const std::shared_ptr<State>& state, int game_type)
{
  assert(player_themesl.size() > 0 && "Please call init(threads) first");

  int level_seed = 0;

  if (USE_LEVEL_SET) {
    int level_index = global_rand_gen.randint(0, NUM_LEVELS);
    level_seed = LEVEL_SEEDS[level_index];
  } else if (NUM_LEVELS > 0) {
    level_seed = global_rand_gen.randint(0, NUM_LEVELS);
  } else {
    level_seed = global_rand_gen.randint();
  }

  RandomMazeGenerator maze_gen;
  maze_gen.rand_gen.seed(level_seed);

  int w = 64;
  int h = 64;
  state->maze.reset(new Maze(w, h, game_type));
  maze_gen.maze = state->maze;

  maze_gen.initial_floor_and_walls(game_type);

  if (game_type==CoinRunPlatforms_v0) {
    maze_gen.generate_coins_on_platforms();
  } else if (game_type == CoinRunToTheRight_v0) {
    maze_gen.generate_coin_to_the_right(game_type);
  } else if (game_type == CoinRunMaze_v0) {
    maze_gen.generate_coin_maze();
  } else {
    fprintf(stderr, "coinrun: unknown game type %i\n", game_type);
    maze_gen.generate_test_level();
  }

  Agent &agent = state->agent;
  float zoom = state->maze->default_zoom;
  agent.maze = state->maze;
  agent.zoom = zoom;
  agent.target_zoom = zoom;

  agent.theme_n = maze_gen.randn(player_themesl.size());
  state->ground_n = maze_gen.randn(ground_themes.size());
  state->bg_n = maze_gen.randn(bg_images.size());

  agent.reset(0);

  state->maze->is_terminated = false;
  state->time = 0;
}

// -- render --

static
int to_shade(float f)
{
  int shade = int(f * 255);
  if (shade < 0) shade = 0;
  if (shade > 255) shade = 255;
  return shade;
}

static
void paint_the_world(
  QPainter& p, const QRect& rect,
  const std::shared_ptr<State>& state, const Agent* agent,
  bool recon, bool lasers)
{
  const_cast<Agent*>(agent)->zoom = 0.9*agent->zoom + 0.1*agent->target_zoom;
  double zoom = agent->zoom;
  const double bgzoom = 0.4;

  bool lowres = rect.height() < 200;
  GroundTheme* ground_theme = choose_ground_theme(state->ground_n, lowres);

  std::shared_ptr<Maze> maze = agent->maze;

  bool maze_render = maze->game_type == CoinRunMaze_v0;

  double kx = zoom * rect.width()  / double(maze->h);  // not w!
  double ky = zoom * rect.height() / double(maze->h);
  double dx = (-agent->x) * kx + rect.center().x()  - 0.5*kx;
  double dy = (agent->y) * ky - rect.center().y()   - 0.5*ky;

  p.setRenderHint(QPainter::Antialiasing, true);
  p.setRenderHint(QPainter::SmoothPixmapTransform, true);
  p.setRenderHint(QPainter::HighQualityAntialiasing, true);

  for (int tile_x=-1; tile_x<=2; tile_x++) {
    for (int tile_y=-1; tile_y<=1; tile_y++) {
      double zx = rect.width()*zoom;   // / bgzoom;
      double zy = rect.height()*zoom;  // / bgzoom);
      QRectF bg_image = QRectF(0, 0, zx, zy);
      bg_image.moveCenter(QPointF(
        zx*tile_x + rect.center().x() + bgzoom*(dx + kx*maze->h/2),
        zy*tile_y + rect.center().y() + bgzoom*(dy - ky*maze->h/2)
        ));

      if (maze_render) {
        p.fillRect(bg_image, QColor(30, 30, 30));
      } else {
        p.drawImage(bg_image, bg_images[state->bg_n]);
      }
    }
  }

  int radius = int(1 + maze->h / zoom);  // actually /2 works except near scroll limits
  int ix = int(agent->x + .5);
  int iy = int(agent->y + .5);
  int x_start = max(ix - radius, 0);
  int x_end = min(ix + radius + 1, maze->w);
  int y_start = max(iy - radius, 0);
  int y_end = min(iy + radius + 1, maze->h);
  double WINH = rect.height();

  for (int y=y_start; y<y_end; ++y) {
    for (int x=x_start; x<x_end; x++) {
      int wkey = maze->get_elem(x, y);
      if (wkey==SPACE) continue;

      auto f = ground_theme->walls.find(wkey);
      QImage img = f == ground_theme->walls.end() ? ground_theme->default_wall : f->second;
      QRectF dst = QRectF(kx*x + dx, WINH - ky*y + dy, kx + .5, ky + .5);
      dst.adjust(-0.1, -0.1, +0.1, +0.1); // here an attempt to fix subpixel seams that appear on the image, especially lowres
      
      if (maze_render) {
        if (is_coin(wkey)) {
          p.fillRect(dst, QColor(255, 255, 0));
        } else if (wkey == WALL_MIDDLE || wkey == WALL_SURFACE) {
          p.fillRect(dst, QColor(150, 150, 150));
        }
      } else if (wkey==LAVA_MIDDLE || wkey==LAVA_SURFACE) {
        QRectF d1 = dst;
        QRectF d2 = dst;
        QRectF sr(QPointF(0,0), img.size());
        QRectF sr1 = sr;
        QRectF sr2 = sr;
        float tr = state->time*0.1;
        tr -= int(tr);
        tr *= -1;
        d1.translate(tr*dst.width(), 0);
        d2.translate(dst.width() + tr*dst.width(), 0);
        sr1.translate(-tr*img.width(), 0);
        sr2.translate(-img.width() - tr*img.width(), 0);
        d1 &= dst;
        d2 &= dst;
        d1.adjust(0, 0, +0.5, 0);
        d2.adjust(-0.5, 0, 0, 0);
        sr1 &= sr;
        sr2 &= sr;
        if (!sr1.isEmpty())
          p.drawImage(d1, img, sr1);
        if (!sr2.isEmpty())
          p.drawImage(d2, img, sr2);
      } else {
        p.drawImage(dst, img);
      }
    }
  }

  PlayerTheme* active_theme = choose_player_theme(agent->theme_n, agent->is_facing_right, lowres);

  if (maze_render) {
    QRectF dst = QRectF(kx * agent->x + dx, WINH - ky * (agent->y+1) + dy + ky, kx, ky);
    p.fillRect(dst, QColor(0, 255, 199));
  } else {
    QImage img = agent->picture(active_theme);
    QRectF dst = QRectF(kx * agent->x + dx, WINH - ky * (agent->y+1) + dy, kx, 2 * ky);
    p.drawImage(dst, img);  
  }

  int monsters_count = maze->monsters.size();
  for (int i=0; i<monsters_count; ++i) {
    const std::shared_ptr<Monster>& m = maze->monsters[i];
    QRectF dst = QRectF(kx*m->x + dx, WINH - ky*m->y + dy, kx, ky);

    EnemyTheme* theme = choose_enemy_theme(m, i, lowres);
    if (m->is_flying || m->is_walking) {
      for (int t=2; t<MONSTER_TRAIL; t+=2) {
        QRectF dst = QRectF(kx*m->prev_x[t] + dx, WINH - ky*m->prev_y[t] + dy, kx, ky);
        float ft = 1 - float(t)/MONSTER_TRAIL;
        float smaller = 0.20;
        float lower = -0.22;
        float soar = -0.4;
        dst.adjust(
          (smaller-0.2*ft)*kx, (soar*ft-0.2*ft-lower+smaller)*ky,
          (-smaller+0.2*ft)*kx, (soar*ft+0.2*ft-lower-smaller)*ky);
        p.setBrush(QColor(255,255,255, t*127/MONSTER_TRAIL));
        p.setPen(Qt::NoPen);
        p.drawEllipse(dst);
      }
    }
    p.drawImage(dst, state->time / theme->anim_freq % 2 == 0 ? theme->walk1 : theme->walk2);
  }

  if (USE_DATA_AUGMENTATION) {
    float max_rand_dim = .25;
    float min_rand_dim = .1;
    int num_blotches = global_rand_gen.randint(0, 6);

    bool hard_blotches = false;

    if (hard_blotches) {
      max_rand_dim = .3;
      min_rand_dim = .2;
      num_blotches = global_rand_gen.randint(0, 10);
    }

    for (int j = 0; j < num_blotches; j++) {
      float rx = global_rand_gen.rand01() * rect.width();
      float ry = global_rand_gen.rand01() * rect.height();
      float rdx = (global_rand_gen.rand01() * max_rand_dim + min_rand_dim) * rect.width();
      float rdy = (global_rand_gen.rand01() * max_rand_dim + min_rand_dim) * rect.height();

      QRectF dst3 = QRectF(rx, ry, rdx, rdy);
      p.fillRect(dst3, QColor(global_rand_gen.randint(0, 255), global_rand_gen.randint(0, 255), global_rand_gen.randint(0, 255)));
    }
  }

  if (PAINT_VEL_INFO) {
    float infodim = rect.height() * .2;
    QRectF dst2 = QRectF(0, 0, infodim, infodim);
    int s0 = to_shade(agent->spring / maze->max_jump);
    int s1 = to_shade(.5 * agent->vx / maze->max_speed + .5);
    int s2 = to_shade(.5 * agent->vy / maze->max_jump + .5);
    p.fillRect(dst2, QColor(s1, s1, s1));

    QRectF dst3 = QRectF(infodim, 0, infodim, infodim);
    p.fillRect(dst3, QColor(s2, s2, s2));
  }
}

// -- vecenv --

class VectorOfStates {
public:
  int game_type;
  int nenvs;
  int handle;
  QMutex states_mutex;
  std::vector<std::shared_ptr<State>> states; // nenvs
};

static QMutex h2s_mutex;
static QWaitCondition wait_for_actions;
static QWaitCondition wait_for_step_completed;
static std::map<int, std::shared_ptr<VectorOfStates>> h2s;
static std::list<std::shared_ptr<State>> workers_todo;
static int handle_seq = 100;

static std::shared_ptr<VectorOfStates> vstate_find(int handle)
{
  QMutexLocker lock(&h2s_mutex);
  auto f = h2s.find(handle);
  if (f == h2s.end()) {
    fprintf(stderr, "cannot find vstate handle %i\n", handle);
    assert(0);
  }
  return f->second;
}

static
void copy_render_buf(int e, uint8_t* obs_rgb, uint8_t* buf, int res_w, int res_h)
{
  for (int y = 0; y<res_h; y++) {
    for (int x = 0; x<res_w; x++) {
      uint8_t* p = obs_rgb + e*res_h*res_w*3 + y*res_w*3 + x*3;

      p[0] = buf[y*res_w*4 + x*4 + 2];
      p[1] = buf[y*res_w*4 + x*4 + 1];
      p[2] = buf[y*res_w*4 + x*4 + 0];
    }
  }
}

static
void paint_render_buf(uint8_t* buf, int res_w, int res_h, const std::shared_ptr<State>& todo_state, const Agent* a, bool recon, bool lasers)
{
  QImage img((uchar*)buf, res_w, res_h, res_w * 4, QImage::Format_RGB32);
  QPainter p(&img);
  paint_the_world(p, QRect(0, 0, res_w, res_h), todo_state, a, recon, lasers);
}

static
void stepping_thread(int n)
{
  while (1) {
    std::shared_ptr<State> todo_state;
    std::list<std::shared_ptr<State>> my_todo;
    while (1) {
      if (shutdown_flag)
        return;
      QMutexLocker sleeplock(&h2s_mutex);
      if (workers_todo.empty()) {
        wait_for_actions.wait(&h2s_mutex, 1000); // milliseconds
        continue;
      }
      my_todo.splice(my_todo.begin(), workers_todo, workers_todo.begin());
      break;
    }
    todo_state = my_todo.front();

    {
      QMutexLocker lock(&todo_state->step_mutex);
      assert(todo_state->agent_ready);
      todo_state->step_in_progress = true;
    }

    {
      QMutexLocker lock(&todo_state->state_mutex);
      std::shared_ptr<VectorOfStates> belongs_to = todo_state->belongs_to.lock();
      if (!belongs_to)
        continue;
      todo_state->time += 1;
      bool game_over = todo_state->maze->is_terminated;

      for (const std::shared_ptr<Monster>& m: todo_state->maze->monsters) {
        m->step(todo_state->maze);
        Agent& a = todo_state->agent;
        if (fabs(m->x - a.x) + fabs(m->y - a.y) < 1.0)
          todo_state->maze->is_terminated = true;  // no effect on agent score
      }

      Agent& a = todo_state->agent;
      if (game_over)
        a.monitor_csv_episode_over();
      a.game_over = game_over;
      a.step(belongs_to->game_type);

      if (game_over) {
        state_reset(todo_state, belongs_to->game_type);
      }

      paint_render_buf(a.render_buf, RES_W, RES_H, todo_state, &a, false, false);
      if (a.render_hires_buf)
        paint_render_buf(a.render_hires_buf, VIDEORES, VIDEORES, todo_state, &a, false, false);
    }

    {
      QMutexLocker lock(&todo_state->step_mutex);
      assert(todo_state->agent_ready);
      assert(todo_state->step_in_progress);
      todo_state->agent_ready = false;
      todo_state->step_in_progress = false;
    }

    wait_for_step_completed.wakeAll();
  }
}

class SteppingThread : public QThread {
public:
  int n;
  SteppingThread(int n)
      : n(n) {}
  void run() { stepping_thread(n); }
};

static std::vector<std::shared_ptr<QThread>> all_threads;

// ------ C Interface ---------

extern "C" {
int get_NUM_ACTIONS()  { return NUM_ACTIONS; }
int get_RES_W()  { return RES_W; }
int get_RES_H()  { return RES_H; }
int get_VIDEORES()  { return VIDEORES; }

void initialize_args(int *int_args) {
  USE_HIGH_DIF = int_args[0] == 1;
  NUM_LEVELS = int_args[1];
  PAINT_VEL_INFO = int_args[2] == 1;
  USE_DATA_AUGMENTATION = int_args[3] == 1;
  DEFAULT_GAME_TYPE = int_args[4];

  int training_sets_seed = int_args[5];
  int rand_seed = int_args[6];

  if (NUM_LEVELS > 0 && (training_sets_seed != -1)) {
    global_rand_gen.seed(training_sets_seed);

    USE_LEVEL_SET = true;

    LEVEL_SEEDS = new int[NUM_LEVELS];

    for (int i = 0; i < NUM_LEVELS; i++) {
      LEVEL_SEEDS[i] = global_rand_gen.randint();
    }
  }

  global_rand_gen.seed(rand_seed);
}

void initialize_set_monitor_dir(const char *d, int monitor_csv_policy_)
{
  monitor_dir = d;
  monitor_csv_policy = monitor_csv_policy_;
}

void init(int threads)
{
  if (bg_images.empty())
    try {
      resource_path = getenv("COINRUN_RESOURCES_PATH");
      if (resource_path == "") {
        throw std::runtime_error("missing environment variable COINRUN_RESOURCES_PATH");
      }
      images_load();
    } catch (const std::exception &e) {
      fprintf(stderr, "ERROR: %s\n", e.what());
      return;
    }

  assert(all_threads.empty());
  all_threads.resize(threads);
  for (int t = 0; t < threads; t++) {
    all_threads[t] = std::shared_ptr<QThread>(new SteppingThread(t));
    all_threads[t]->start();
  }
}

int vec_create(int game_type, int nenvs, int lump_n, bool want_hires, float default_zoom)
{
  std::shared_ptr<VectorOfStates> vstate(new VectorOfStates);
  vstate->states.resize(nenvs);
  vstate->game_type = game_type;

  for (int n = 0; n < nenvs; n++) {
    vstate->states[n] = std::shared_ptr<State>(new State(vstate));
    vstate->states[n]->state_n = n;
    state_reset(vstate->states[n], vstate->game_type);
    vstate->states[n]->agent_ready = false;
    vstate->states[n]->agent.zoom = default_zoom;
    vstate->states[n]->agent.target_zoom = default_zoom;
    if (
        (monitor_csv_policy == 1 && n == 0) ||
        (monitor_csv_policy == 2))
    {
      vstate->states[n]->agent.monitor_csv_open(n + lump_n * nenvs);
    }
    if (want_hires)
        vstate->states[n]->agent.render_hires_buf = new uint8_t[VIDEORES*VIDEORES*4];
  }
  vstate->nenvs = nenvs;
  int h;
  {
    QMutexLocker lock(&h2s_mutex);
    h = handle_seq++;
    h2s[h] = vstate;
    vstate->handle = h;
  }
  return h;
}

void vec_close(int handle)
{
  if (handle == 0)
    return;
  std::shared_ptr<VectorOfStates> vstate = vstate_find(handle);
  {
    QMutexLocker lock(&h2s_mutex);
    h2s.erase(handle);
  }
}

void vec_step_async_discrete(int handle, int32_t *actions)
{
  std::shared_ptr<VectorOfStates> vstate = vstate_find(handle);
  QMutexLocker sleeplock(&h2s_mutex);
  {
    QMutexLocker lock2(&vstate->states_mutex);
    for (int e = 0; e < vstate->nenvs; e++) {
      std::shared_ptr<State> state = vstate->states[e];
      assert((unsigned int)actions[e] < (unsigned int)NUM_ACTIONS);
      state->agent.action_dx = DISCRETE_ACTIONS[2 * actions[e] + 0];
      state->agent.action_dy = DISCRETE_ACTIONS[2 * actions[e] + 1];
      {
        QMutexLocker lock3(&state->step_mutex);
        state->agent_ready = true;
        workers_todo.push_back(state);
      }
    }
  }
  wait_for_actions.wakeAll();
}

void vec_wait(
  int handle,
  uint8_t* obs_rgb,
  uint8_t* obs_hires_rgb,
  float* rew,
  bool* done)
{
  std::shared_ptr<VectorOfStates> vstate = vstate_find(handle);
  while (1) {
    QMutexLocker sleeplock(&h2s_mutex);
    bool all_steps_completed = true;
    {
      QMutexLocker lock2(&vstate->states_mutex);
      for (int e = 0; e < vstate->nenvs; e++) {
        std::shared_ptr<State> state = vstate->states[e];
        QMutexLocker lock3(&state->step_mutex);
        all_steps_completed &= !state->agent_ready;
      }
    }
    if (all_steps_completed)
      break;
    wait_for_step_completed.wait(&h2s_mutex, 1000); // milliseconds
  }
  QMutexLocker lock1(&vstate->states_mutex);
  for (int e = 0; e < vstate->nenvs; e++) {
    std::shared_ptr<State> state_e = vstate->states[e];
    QMutexLocker lock2(&state_e->state_mutex);
    // don't really need a mutex, because step is completed, but it's cheap to lock anyway
    Agent& a = state_e->agent;
    if (a.render_hires_buf)
      copy_render_buf(e, obs_hires_rgb, a.render_hires_buf, VIDEORES, VIDEORES);
    copy_render_buf(e, obs_rgb, a.render_buf, RES_W, RES_H);

    rew[e] = a.reward;
    done[e] = a.game_over;
    a.reward = 0;
    a.game_over = false;
  }
}

void coinrun_shutdown()
{
  shutdown_flag = true;
  while (!all_threads.empty()) {
    std::shared_ptr<QThread> th = all_threads.back();
    all_threads.pop_back();
    th->wait();
    assert(th->isFinished());
  }
}
}

// ------------ GUI -------------

class Viz : public QWidget {
public:
  std::shared_ptr<VectorOfStates> vstate;
  std::shared_ptr<State> show_state;
  int control_handle = -1;

  int font_h;
  int render_mode = 0;
  bool recon = false;
  bool lasers = false;

  void paint(QPainter& p, const QRect& rect)
  {
    Agent& agent = show_state->agent;

    paint_the_world(p, rect, show_state, &agent, recon, lasers);

    QRect text_rect = rect;
    text_rect.adjust(font_h/3, font_h/3, -font_h/3, -font_h/3);
    p.drawText(text_rect, Qt::AlignRight|Qt::AlignTop, QString::fromStdString(std::to_string(agent.time_alive)));
  }

  void paintEvent(QPaintEvent *ev)
  {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::SmoothPixmapTransform, true);
    p.setRenderHint(QPainter::HighQualityAntialiasing, true);

    if (render_mode==0) {
      QRect r = rect();
      paint(p, r);

    } else if (render_mode>0) {
      QPixmap bm(render_mode, render_mode);
      {
        QPainter p2(&bm);
        p2.setFont(font());
        paint(p2, QRect(0, 0, render_mode, render_mode));
      }
      p.drawPixmap(rect(), bm);
    }

    if (ffmpeg.isOpen()) {
      QByteArray txt1 = ffmpeg.readAllStandardError();
      if (!txt1.isEmpty())
        fprintf(stderr, "ffmpeg stderr %s\n", txt1.data());
      QImage img(VIDEORES, VIDEORES, QImage::Format_RGB32);
      {
        QPainter p2(&img);
        p2.setFont(font());
        paint(p2, QRect(0, 0, VIDEORES, VIDEORES));
      }
      ffmpeg.write((char*)img.bits(), VIDEORES*VIDEORES*4);
    }
  }

  void set_render_mode(int m)
  {
    render_mode = m;
    if (render_mode==0) {
      choose_font(rect().height());
    }
    if (render_mode>0) {
      choose_font(render_mode);
    }
  }

  QProcess ffmpeg;

  void resizeEvent(QResizeEvent *ev) {
    choose_font(ev->size().height());
    QWidget::resizeEvent(ev);
  }

  void choose_font(int resolution_h) {
    int h = std::min(std::max(resolution_h / 20, 10), 100);
    QFont f("Courier");
    f.setPixelSize(h);
    setFont(f);
    font_h = h;
  }
};

int convert_action(int dx, int dy) {
  if (dy == -1) {
    return NUM_ACTIONS - 1;
  }

  for (int i = 0; i < NUM_ACTIONS; i++) {
    if (dx == DISCRETE_ACTIONS[2 * i] && dy == DISCRETE_ACTIONS[2 * i + 1]) {
      return i;
    }
  }

  assert(false);

  return 0;
}

class TestWindow : public QWidget {
  Q_OBJECT
public:
  QVBoxLayout *vbox;
  Viz *viz;

  TestWindow()
  {
    viz = new Viz();
    vbox = new QVBoxLayout();
    vbox->addWidget(viz, 1);
    setLayout(vbox);
    actions[0] = 0;
    startTimer(66);
  }

  ~TestWindow()
  {
    delete viz;
  }

  void timeout()
  {
    vec_step_async_discrete(viz->control_handle, actions);
    uint8_t bufrgb[RES_W * RES_H * 3];
    float bufrew[1];
    bool bufdone[1];
    vec_wait(viz->control_handle, bufrgb, 0, bufrew, bufdone);
    // fprintf(stderr, "%+0.2f %+0.2f %+0.2f\n", bufvel[0], bufvel[1], bufvel[2]);
  }

  int rolling_state_update = 0;

  std::map<int, int> keys_pressed;

  void keyPressEvent(QKeyEvent* kev)
  {
    keys_pressed[kev->key()] = 1;
    if (kev->key() == Qt::Key_Return) {
      viz->show_state->maze->is_terminated = true;
    }
    if (kev->key() == Qt::Key_R) {
  if (viz->ffmpeg.isOpen()) {
    fprintf(stderr, "finishing rec\n");
    viz->ffmpeg.closeWriteChannel();
    viz->ffmpeg.waitForFinished();
    fprintf(stderr, "finished rec\n");
  } else {
    fprintf(stderr, "starting ffmpeg\n");
    QStringList arguments;
    arguments << "-y" << "-r" << "30" <<
      "-f" << "rawvideo" << "-s:v" << (VIDEORES_STR "x" VIDEORES_STR) << "-pix_fmt" << "rgb32" <<
      "-i" << "-" << "-vcodec" << "libx264" << "-pix_fmt" << "yuv420p" << "-crf" << "10" <<
      "coinrun-manualplay.mp4";
    viz->ffmpeg.start("ffmpeg", arguments);
    bool r = viz->ffmpeg.waitForStarted();
    fprintf(stderr, "video rec started %i\n", int(r));
  }
    }
    if (kev->key() == Qt::Key_F1)
      viz->set_render_mode(0);
    if (kev->key() == Qt::Key_F2)
      viz->set_render_mode(64);
    if (kev->key() == Qt::Key_F5)
      viz->show_state->agent.target_zoom = 1.0;
    if (kev->key() == Qt::Key_F6)
      viz->show_state->agent.target_zoom = 2.0;
    if (kev->key() == Qt::Key_F7)
      viz->show_state->agent.target_zoom = 3.0;
    if (kev->key() == Qt::Key_F8)
      viz->show_state->agent.target_zoom = 5.0;

    if (kev->key() == Qt::Key_F9)
      viz->recon ^= true;
    if (kev->key() == Qt::Key_F10)
      viz->lasers ^= true;

    control();
  }

  void keyReleaseEvent(QKeyEvent *kev)
  {
    keys_pressed[kev->key()] = 0;
    control();
  }

  int actions[1];

  void control()
  {
    if (viz->control_handle <= 0)
      return;
    int dx = keys_pressed[Qt::Key_Right] - keys_pressed[Qt::Key_Left];
    int dy = keys_pressed[Qt::Key_Up] - keys_pressed[Qt::Key_Down];

    actions[0] = convert_action(dx, dy);
  }

  void timerEvent(QTimerEvent *ev)
  {
    timeout();
    update();
    update_window_title();
  }

  void update_window_title()
  {
    setWindowTitle(QString::fromUtf8(
    stdprintf("CoinRun game_type=%i zoom=%0.2f res=%ix%i",
      viz->vstate->game_type,
      viz->show_state->agent.zoom,
      viz->render_mode, viz->render_mode
      ).c_str()));
  }
};

extern "C" void test_main_loop()
{
  QApplication *app = 0;
  TestWindow *window = 0;
#ifdef Q_MAC_OS
  [NSApp activateIgnoringOtherApps:YES];
#endif
  {
    static int argc = 1;
    static const char *argv[] = {"CoinRun"};
    app = new QApplication(argc, const_cast<char **>(argv));
  }

  int handle = vec_create(DEFAULT_GAME_TYPE, 1, 0, false, 5.0);

  window = new TestWindow();
  window->resize(800, 800);

  window->viz->control_handle = handle;
  {
    std::shared_ptr<VectorOfStates> vstate = vstate_find(handle);
    window->viz->vstate = vstate;
    window->viz->show_state = vstate->states[0];
  }
  window->show();

  app->exec();

  delete window;
  delete app;

  vec_close(handle);
  coinrun_shutdown();
}

#include ".generated/coinrun.moc"
