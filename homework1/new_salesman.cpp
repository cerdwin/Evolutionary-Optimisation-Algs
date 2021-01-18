#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <queue>
#include <fstream>

#define ALGORITHM 1 // 1- genetic algorithm, 2 - simulated annealing, 3 - (memetic) genetic algorithm employing nearest neighbour heuristic for initial population, 4 - Nearest neighbour constructive heuristic
#define POPULATION_SIZE 100
#define IMPROVEMENTS 100
#define MEMETIC_ALGORITHM 1 // 1 - ga with improved initial population, 2 - ga with improved "elite" population selected, 3 - improved offspring population 4 - original genetic algorithm
#define GENERATIONS 100
#define ELITE_SELECTION_FRACTION 0.05
#define SWAPS 1
#define FLIP_SEGMENT_SIZE 1/8
#define TRANSLOCATION_SEGMENT_FRACTION 1/8
#define TEMPERATURE 1000
#define SA_ITERATIONS 1000
#define ALPHA 0.88
#define PERTURBATION 1 // 1 - swap perturbation, 2 - flip perturbation, 3 - 
#define CROSS_OVER 1 // 1 - circle cross-over, 2 - nwox_crossover

using namespace std;
 
int towns;
int *** travel_plan;
typedef struct{
    int id;
    int x_pos;
    int y_pos;
    
}town_t;



typedef struct path_t{
    int cost;
    int id;
    std::vector<int> path;
    friend bool operator<(const path_t& a, const path_t& b){
        return a.cost > b.cost;
    }
  
}path_t;

struct comparator{
    bool operator()(const path_t * a, const path_t * b)
    {
        return a->cost > b->cost;
    }
};

town_t ** town_array;
int ** distance_matrix;
std::vector<vector<int> > result;

//////////////////////////////////////// EVALUATION ////////////////////////////////////////////////////////
int evaluate(std::vector<int> trip){
    int result = 0;
    for(int i = 0; i< towns; i++){
        result+=distance_matrix[trip[i]-1][trip[i+1]-1];
    }
    return result;
}

/////////////////////////////////////// HELPER FUNCTIONS //////////////////////////////////////////////////

std::vector<int> middle_cities(int avoided){
    /*returns a random combination of cities without the start/end city*/
    int index = 0;
    std::vector<int> result;
    for(int i = 1; i <= towns; i++){
        if(i != avoided){
            result.push_back(i);
        }
    }
    std::random_shuffle ( result.begin(), result.end() );
    return result;
    
}


void print_matrix(){
    /*self-explanatory, returns the matrix of euclidean distances between cities */
    for(int i = 0; i < towns; i++){
        for(int x = 0; x < towns; x++){
            printf("%d ",distance_matrix[i][x] );
        }
        printf("\n");
    }
}

std::vector<int> random_solution(){
    /*creates a random path between cities*/
    std::vector<int> result;
    int start_town = rand() % towns +1;
    result.push_back(start_town);
    std::vector<int> temp = middle_cities(start_town);
    result.insert(result.end(), temp.begin(), temp.end());
    result.push_back(start_town);
    
    int grade = evaluate(result);
    

    return result;
}



int eucdist(town_t * first, town_t * second){
    /*Function returning euclidean distance between two cities*/
    double x = first->x_pos - second->x_pos; 
    double y = first->y_pos - second->y_pos;
    double distance = pow(x, 2) + pow(y, 2);       
    return sqrt(distance); 
}

void load_data(){
    /**
    A function responsible for loading of data. I have stored input in .in files, where first line states
    the number of cities and resulting lines in the form of A B C represent a start town (A), destination (B)
    and cost incurred by traversing the distance between them (C)

    **/

    scanf("%d\n", &towns);
    town_array = (town_t**)calloc(towns,sizeof(town_t*));
    for(int i = 0; i < towns; i++){
        town_t * temp_town = (town_t*)calloc(1, sizeof(town_t));
        temp_town->id =  temp_town->x_pos = temp_town->y_pos =  0;
        town_array[i] = temp_town;
        
    }
    int x = 0;
    int id = 0;
    int y = 0;
    for(int i = 0; i < towns; i++){
        scanf("%d %d %d", &id, &x, &y);
        town_array[i]->id = id-1; // pozor, toto se musi pak opravit, aby sedel vysledek
        town_array[i]->x_pos = x;
        town_array[i]->y_pos = y;

    }
    distance_matrix = (int**)calloc(towns, sizeof(int*));
    for(int i = 0; i < towns; i++){
        distance_matrix[i] = (int*)calloc(towns, sizeof(int));
    }
    for(int i = 0; i < towns; i++){
        for(int x = 0; x < towns; x++){
            distance_matrix[i][x] = eucdist(town_array[i], town_array[x]);
        }
    }
}

/////////////////////////////////// PERTURBATIONS //////////////////////////////////////////////

std::vector<int> swap_perturbation(std::vector<int> path){
    int temp;
    for(int i = 0; i < SWAPS; i++){
        int first = rand() % path.size();
        int second = rand()% path.size();
        temp = path[first];
        path[first] = path[second];
        path[second] = temp;
    }
    return path;
}

std::vector<int> flip_perturbation(std::vector<int> path){
    int flip_size = path.size()*FLIP_SEGMENT_SIZE;
    int first_random = random()% path.size();
    int second_random = random()% (path.size()-first_random);
    std::reverse(path.begin()+first_random,path.begin()+first_random+second_random);

    return path;
}

std::vector<int> perturbate(std::vector<int> path){
    std::vector<int> result;
    std::vector<int> shorter_path;
    shorter_path.assign(path.begin()+1, path.end()-1);
    
    if(PERTURBATION == 1){
        result = swap_perturbation(shorter_path);
    }else if(PERTURBATION == 2){
        result = flip_perturbation(shorter_path);
    }else if(PERTURBATION == 3){
       // result = translocation_perturbation(shorter_path);
    }

    std::vector<int> ret;
    ret.insert(ret.end(), result.begin(), result.end());
    ret.insert(ret.begin(), path.begin(), path.begin()+1);
    ret.push_back(path[0]);

    return ret;
}
/////////////////////////////////////// CROSS-OVER //////////////////////////////////////////////////
std::vector<int> cycle_crossover(std::vector<int> father, std::vector<int> mother, int child_index){
    std::vector<int> father_copy;
    std::vector<int> first_child;
    std::vector<int> second_child;
    
      
    int father_size = sizeof(father);
    for(int i = 0; i < father.size(); i++){
        father_copy.push_back(0);
        first_child.push_back(0);
        second_child.push_back(0);
    }

    int cycle = 0;
    
    for(int i = 0; i < father.size(); i++){
        if(father_copy[i] == -1)continue;

        int head = father[i];
        //cout << "Head is: ";
        //cout << head;
        //cout << "\n";
        int current = father[i];
        bool started = false;
        while(current != head || !started){
            if(!started){
              started = true;
            }
            int index = find(father.begin(), father.end(), current) - father.begin(); // 
            //cout << "Index is:" << index << "\n";

            if(cycle % 2 == 0){
              first_child[index] = father[index];
              second_child[index] = mother[index];
            }else{
              second_child[index] = father[index];
              first_child[index] = mother[index];
            }
            current = mother[index];
            father_copy[index] = -1;
        }
        //cout << "First cycle complete\n";
        //printf("First child:\n");
        //print(first_child);
        //printf("\nSecond child:\n");
        //print(second_child);
        //cout << "\n";
        cycle++;
    }
    //printf("got here\n");
    first_child.back() = first_child.front();
    second_child.back() = second_child.front();
    
    if(child_index == 1){
        return first_child;
    }
    return second_child;
}

std::vector<int> nwox_crossover(std::vector<int> father, std::vector<int> mother, int child_index){
    std::vector<int>first_child;
    std::vector<int>second_child;
    for(int i = 0; i < father.size(); i++){
        first_child.push_back(0);
        second_child.push_back(0);
    }
    int a = 1;
    int b = 1;
    while(a >= b ){
        a = rand() % father.size();
        b = rand() % father.size();
    }
    //cout << a << " " << b << "\n";
    for(int i = 0; i < father.size(); i++){
 
      std::vector<int>::iterator finder = std::find(mother.begin()+a, mother.begin()+b, father[i]);
      if(finder == mother.begin()+b){
          first_child[i] = father[i];
      }else{
        first_child[i] = 0;
      }
      finder = std::find(father.begin()+a, father.begin()+b, mother[i]);
      if(finder == father.begin()+b){
          second_child[i] = mother[i];
      }else{
        second_child[i] = 0;
      }
        
    }
    /*
    cout << "first stage:";
    cout<< '\n';
    printf("First child:\n");

    for(int i = 0; i < first_child.size(); i++){
        printf("%d ",first_child[i] );
    }
    printf("\n");
    cout<< '\n';
    printf("Second child:\n");
    for(int i = 0; i < second_child.size(); i++){
        printf("%d ",second_child[i] );
    }
    printf("\n");
    */
    
    std::vector<int>temp1;
    std::vector<int>temp2;
    for(int i = 0; i < father.size(); i++){
        temp1.push_back(0);
        temp2.push_back(0);
    }
    int first_index = 0;
    int second_index = 0;
    for(int i = 0; i < father.size(); i++){
      if(i == a+(b-a)/2){
        // fill with the interval between
        // fill with the interval between
        for(int i = a; i <b; i++){
          temp1[first_index++] = 0;
          temp2[second_index++] = 0;
        }
      }
      if(first_child[i]!= 0){
        temp1[first_index++] = first_child[i];
      }
      if(second_child[i]!= 0){
        temp2[second_index++] = second_child[i];
      }
    }
    
    /*
    cout << "second stage:";
    printf("First temp:\n");
    for(int i = 0; i < temp1.size(); i++){
        printf("%d ",temp1[i] );
    }
    printf("\n");
    cout<< '\n';
    printf("Second temp:\n");
    for(int i = 0; i < temp2.size(); i++){
        printf("%d ",temp2[i] );
    }
    printf("\n");    
    cout<< '\n';
    */
    first_index = 0;
    second_index = 0;
    for(int i = 0; i < father.size(); i++){
      if(temp1[i] != 0){
        first_child[i] = temp1[i];
        second_child[i] = temp2[i];
      }else{
        first_child[i] = mother[i];
        second_child[i] = father[i];
      }
    }
    
  
    cout << "third stage:";
    printf("First child:\n");
    for(int i = 0; i < first_child.size(); i++){
        printf("%d ",first_child[i] );
    }
    printf("\n");
    cout<< '\n';
    printf("Second child:\n");
    for(int i = 0; i < second_child.size(); i++){
        printf("%d ",second_child[i] );
    }
    printf("\n");
    cout<< '\n' << "Size of children:" << second_child.size() << "\n";
  
    if(child_index == 1){
        return first_child;
    }
  return second_child;
}

std::vector<int> cross_over(std::vector<int> father, std::vector<int> mother, int child_index){
    if(CROSS_OVER == 1){
        return cycle_crossover(father, mother, child_index);

    }else if(CROSS_OVER == 2){
        return nwox_crossover(father, mother,child_index);

    }
    return father;
}
///////////////////////////////////////// SIMULATED ANNEALING //////////////////////////////////////////////
std::vector<int> neighbouring_solution(std::vector<int> path){
    int a = random() % path.size();
    int b = random() % path.size();
    while(a == b){
        b = random() % path.size();
    }
    int item_a = path[a];
    path[a] = path[b];
    path[b] = item_a;
    return path;
    
}

int SA(std::vector<int> trip){
    double temperature = TEMPERATURE;
    // generate a neighbouring solution
    int best_cost = evaluate(trip);
    printf("best cost at start:%d\n",best_cost );
    std::vector<int> new_candidate;
    for(int i = 0; i < SA_ITERATIONS; i++){
        new_candidate = neighbouring_solution(trip);
        int new_cost = evaluate(new_candidate);
        //printf("new cost:%d\n",new_cost );
        //printf("temperature:%f\n", temperature);
        if(new_cost<best_cost){
            trip = new_candidate;
            best_cost = new_cost;
        }else {
            double p = exp(-(new_cost-best_cost)/temperature);
            //double p = (random()%100);
            double r = (random()%100);
            if(r < p){
                trip = new_candidate;
                best_cost = new_cost;
            }
        }
        //temperature = pow(ALPHA, i)*temperature;
        temperature = 0.988*temperature;
    }
    cout << "Best trip consists of:\n";
    for(int i = 0; i < trip.size(); i++){
        printf("%d ", trip[i]);
    }
    printf("\n");
    printf("best cost now:%d\n",best_cost );
    //printf("length of trip:%d\n",trip.size() );    
    return 0;
}
/////////////////////////////////////// NEAREST NEIGHBOUR CONSTRUCTIVE HEURISTIC //////////////////////////////////////////////////
void nearest_neighbour(int starting_position){
    std::vector<int> resulting_path;
    std::vector<int> visited;
    for(int i = 0; i < towns; i++){
        visited.push_back(-1);
    }
    visited[starting_position] = 1;
    resulting_path.push_back(starting_position);
    int shortest_distance = INT_MAX;
    int closest_city = -1;
    for(int i = 0; i < towns-1; i++ ){
        for(int i = 0; i < towns; i++){
            if(visited[i] == -1 && distance_matrix[starting_position][i]<shortest_distance){
                shortest_distance = distance_matrix[starting_position][i];
                closest_city = i;

            }
        }
        visited[closest_city] = 1;
        shortest_distance = INT_MAX;
        int current_neighbour = closest_city+1;
        resulting_path.push_back(current_neighbour);
        starting_position = closest_city;
        closest_city = -1;
    }
    resulting_path.push_back(0);
    resulting_path.back() = resulting_path.front();
    cout<< "Nearest-neighbour path contains:\n";
    for(int i = 0; i < resulting_path.size(); i++){
        printf("%d ", resulting_path[i]);
    }
    printf("\ncost:%d\n",evaluate(resulting_path) );
    /*printf("\n");
    printf("length of trip:%d\n",resulting_path.size() );    
    //int result = evaluate(resulting_path);
    std::vector<int> rewrite = random_solution();
    printf("solution:\n");
    for(int i = 0; i < resulting_path.size(); i++){
        printf("%d ", resulting_path[i]);
    }
    printf("\n");
    printf("random:\n");
    for(int i = 0; i< rewrite.size(); i++){
        printf("%d ",rewrite[i] );
    }
    printf("\n");*/
    
}

std::vector<int> nearest_changed(int starting_position){
    std::vector<int> resulting_path;
    std::vector<int> visited;
    for(int i = 0; i < towns; i++){
        visited.push_back(-1);
    }
    visited[starting_position] = 1;
    resulting_path.push_back(starting_position);
    int shortest_distance = INT_MAX;
    int closest_city = -1;
    for(int i = 0; i < towns-1; i++ ){
        for(int i = 0; i < towns; i++){
            if(visited[i] == -1 && distance_matrix[starting_position][i]<shortest_distance){
                shortest_distance = distance_matrix[starting_position][i];
                closest_city = i;

            }
        }
        visited[closest_city] = 1;
        shortest_distance = INT_MAX;
        int current_neighbour = closest_city+1;
        resulting_path.push_back(current_neighbour);
        starting_position = closest_city;
        closest_city = -1;
    }
    resulting_path.push_back(0);
    resulting_path.back() = resulting_path.front();
    cout<< "Nearest-neighbour path contains:\n";
    for(int i = 0; i < resulting_path.size(); i++){
        printf("%d ", resulting_path[i]);
    }
    printf("\ncost:%d\n",evaluate(resulting_path) );
    return resulting_path;
    /*printf("\n");
    printf("length of trip:%d\n",resulting_path.size() );    
    //int result = evaluate(resulting_path);
    std::vector<int> rewrite = random_solution();
    printf("solution:\n");
    for(int i = 0; i < resulting_path.size(); i++){
        printf("%d ", resulting_path[i]);
    }
    printf("\n");
    printf("random:\n");
    for(int i = 0; i< rewrite.size(); i++){
        printf("%d ",rewrite[i] );
    }
    printf("\n");*/
    
}
/////////////////////////////////////// MEMETIC GENETIC ALGORITHM //////////////////////////////////////////////////
std::vector<int> not_so_random_path(int starting_position){
    std::vector<int> resulting_path;
    std::vector<int> visited;
    for(int i = 0; i < towns; i++){
        visited.push_back(-1);
    }
    visited[starting_position] = 1;
    resulting_path.push_back(starting_position);
    int shortest_distance = INT_MAX;
    int closest_city = -1;
    for(int i = 0; i < towns-1; i++ ){
        for(int i = 0; i < towns; i++){
            if(visited[i] == -1 && distance_matrix[starting_position][i]<shortest_distance){
                shortest_distance = distance_matrix[starting_position][i];
                closest_city = i;

            }
        }
        visited[closest_city] = 1;
        shortest_distance = INT_MAX;
        int current_neighbour = closest_city+1;
        resulting_path.push_back(current_neighbour);
        starting_position = closest_city;
        closest_city = -1;
    }
    resulting_path.push_back(0);
    resulting_path.back() = resulting_path.front();
    return resulting_path;
}
std::vector<int> improve_solution(){
    std::srand ( unsigned ( std::time(0) ) );
    std::vector<int>  solution = random_solution();
    int current_cost = evaluate(solution);
    for(int i = 0; i< IMPROVEMENTS; i++){
        std::vector<int>  new_solution = perturbate(solution);
        int new_cost = evaluate(new_solution);
        if( new_cost<current_cost){
            solution = new_solution;
            current_cost = new_cost;
        }
    }
    return solution;
}
std::vector<int> improvement_helper(std::vector<int> solution, int extent){
    if(MEMETIC_ALGORITHM == 1){
        if(extent == 1){return improve_solution();}
    }else if(MEMETIC_ALGORITHM == 2){
        if(extent == 2){
            std::vector<int> current = solution;
            int current_cost = evaluate(current);
            for(int i = 0; i < IMPROVEMENTS; i++){
                std::vector<int>new_solution = perturbate(solution);
                int new_cost = evaluate(new_solution);
                if(new_cost<current_cost){
                    solution = new_solution;
                    current_cost = new_cost;
                }
            }
            return solution;
        }
       // cout << "This is invalid input\n";

    }else if(MEMETIC_ALGORITHM == 3){
        if(extent == 3){
            std::vector<int> current = solution;
            int current_cost = evaluate(current);
            for(int i = 0; i < IMPROVEMENTS; i++){

                std::vector<int>new_solution = perturbate(solution);

                int new_cost = evaluate(new_solution);
                if(new_cost<current_cost){
                    solution = new_solution;
                    current_cost = new_cost;
                }
            }
            return solution;
        }

    }
    return solution;
}


void memetic_genetic_algorithm(){
    std::vector<vector<int> > population;
    std::priority_queue<path_t*,vector<path_t*>,comparator> path_queue;

    // 1. First we generate a random acceptable solution to the TSP in the size of the entire population
    for(int i = 0; i < POPULATION_SIZE; i++){
        int start = random()%towns;
        std::vector<int> solution = improvement_helper(random_solution(), 1);
        
        path_t * temp_path = (path_t*)calloc(1, sizeof(path_t));
        temp_path->id = i;
        int cost = evaluate(random_solution());
        temp_path->cost = cost;
        temp_path->path = solution;

        population.push_back(solution);

        path_queue.push(temp_path);

    }
    for(int i = 0; i < GENERATIONS;i++){
        std::priority_queue<path_t*,vector<path_t*>,comparator> copy_queue = path_queue;
        std::priority_queue<path_t*,vector<path_t*>,comparator> selection_queue;
        path_t * curr;
        // 2. WE SELECT THE BEST 20percent of generated paths and add them to our queue - Elitism Selection
        //printf("selecting our elite\n");
        for(int i = 0; i < ELITE_SELECTION_FRACTION*POPULATION_SIZE; i++){
            curr = copy_queue.top();
            curr->path = improvement_helper(curr->path, 2);
            selection_queue.push(curr);
            copy_queue.pop();
        }
        // 2.1 WE FILL the rest of the new population with members of the old population with a certain probability that makes it so the more successful ones have a higher chance
        // we empty the copy queue
        copy_queue = path_queue;
        int descending_variable = POPULATION_SIZE;
        int counter = ELITE_SELECTION_FRACTION*POPULATION_SIZE;
        while(counter<POPULATION_SIZE){
            if(copy_queue.size() == 0){
                copy_queue = path_queue;
            }
            curr = copy_queue.top();
            copy_queue.pop();
            int random_variable = random() % POPULATION_SIZE;
            if(random_variable < descending_variable){
                selection_queue.push(curr);
                counter++;
            }
            if(descending_variable-4 < 0)descending_variable = POPULATION_SIZE;
            descending_variable-=4;

        }


        // 3. We cross members of the population and thus make a new one
        //cout << "ready to cross\n";
        // 3.1. First we virtually double the current population and "shuffle it" to make a pool of parents
        copy_queue = selection_queue;
        std::queue<path_t*> breeding_population;
        counter = 0;
        while(counter < POPULATION_SIZE){
            int random_variable = random()% 10;
            if(random_variable< 4){
                breeding_population.push(copy_queue.top());
                copy_queue.pop();
                counter++;
            }
            if(copy_queue.size() == 0){
                copy_queue = selection_queue;
            }
        }
        /*
                printf("UNLOADING//////////\n");
        int x = 0;
        while(!breeding_population.empty()){
            path_t * temp = breeding_population.front();
            breeding_population.pop();
            printf("///////////%d\n",x++ );
            for(int i = 0; i < temp->path.size(); i++){
                printf("%d ",temp->path[i] );
            }
            printf("\n");
        }

        printf("///////////\n");
        return;

        */

        
        // 3.2. Second, we take two parents and make an offspring
        //cout << "now we have created our parent population, we're off to make babies\n";
        
        while(!path_queue.empty()){
            path_queue.pop();
        }
        //cout << "finished popping\n";
        std::vector<int > first_kid;
        std::vector<int > second_kid;
        while(!breeding_population.empty()){
            path_t * father = breeding_population.front();
            breeding_population.pop();
            if(breeding_population.empty())break;

            path_t * mother = breeding_population.front();
            breeding_population.pop();
            /*cout << "father:\n";
            for(int i = 0; i < father->path.size(); i++){
                printf("%d ", father->path[i]);
            }
            printf("\nmother:\n");
            for(int i = 0; i < mother->path.size(); i++){
                printf("%d ", mother->path[i]);
            }
            printf("\n");
            */

            first_kid = cross_over(father->path, mother->path, 1);
            first_kid = improvement_helper(first_kid, 3);
            /*printf("try to be here\n");
            for(int i = 0; i < first_kid.size(); i++){
                printf("%d ", first_kid[i]);
            }
            printf("\n");*/
            second_kid =cross_over(father->path, mother->path, 2);
            second_kid = improvement_helper(second_kid, 3);
            //cout << "we have finished breeding\n";

            // place children into the new population
            path_t * first_offspring = (path_t*)calloc(1, sizeof(path_t));
            path_t * second_offspring = (path_t*)calloc(1, sizeof(path_t));
            
            first_offspring->id = father->id;
            second_offspring->id = mother->id;
            //printf("so far so good...\n");
            //printf("evaluate first:%d, evaluate second:%d\n",evaluate(first_kid), evaluate(second_kid) );
            first_offspring->cost = evaluate(first_kid);
            second_offspring->cost = evaluate(second_kid);

            first_offspring->path = first_kid;
            second_offspring->path = second_kid;

            //population.push_back(first_child);
            //population.push_back(second_child);
            path_queue.push(first_offspring);
            path_queue.push(second_offspring);

        }
        printf("current generation:%d, best path cost:%d\n",i, path_queue.top()->cost  );
        result.push_back(path_queue.top()->path);
    }

    return;
}

///////////////////////////////////////  GENETIC ALGORITHM //////////////////////////////////////////////////

void genetic_algorithm(){
    std::vector<vector<int> > population;
    std::priority_queue<path_t*,vector<path_t*>,comparator> path_queue;

    // 1. First we generate a random acceptable solution to the TSP in the size of the entire population
    for(int i = 0; i < POPULATION_SIZE; i++){
        std::vector<int> solution = random_solution();
        path_t * temp_path = (path_t*)calloc(1, sizeof(path_t));
        temp_path->id = i;
        temp_path->cost = evaluate(solution);
        temp_path->path = solution;
        population.push_back(solution);
        path_queue.push(temp_path);

    }


    for(int i = 0; i < GENERATIONS;i++){
        std::priority_queue<path_t*,vector<path_t*>,comparator> copy_queue = path_queue;
        std::priority_queue<path_t*,vector<path_t*>,comparator> selection_queue;
        path_t * curr;
        // 2. WE SELECT THE BEST 20percent of generated paths and add them to our queue - Elitism Selection
        for(int i = 0; i < ELITE_SELECTION_FRACTION*POPULATION_SIZE; i++){
            curr = copy_queue.top();
            selection_queue.push(curr);
            copy_queue.pop();
        }
        // 2.1 WE FILL the rest of the new population with members of the old population with a certain probability that makes it so the more successful ones have a higher chance
        // we empty the copy queue
        copy_queue = path_queue;
        int descending_variable = POPULATION_SIZE;
        int counter = ELITE_SELECTION_FRACTION*POPULATION_SIZE;
        while(counter<POPULATION_SIZE){
            if(copy_queue.size() == 0){
                copy_queue = path_queue;
            }
            curr = copy_queue.top();
            copy_queue.pop();
            int random_variable = random() % POPULATION_SIZE;
            if(random_variable < descending_variable){
                selection_queue.push(curr);
                counter++;
            }
            if(descending_variable-4 < 0)descending_variable = POPULATION_SIZE;
            descending_variable-=4;

        }


        // 3. We cross members of the population and thus make a new one
        //cout << "ready to cross\n";
        // 3.1. First we virtually double the current population and "shuffle it" to make a pool of parents
        copy_queue = selection_queue;
        std::queue<path_t*> breeding_population;
        counter = 0;
        while(counter < POPULATION_SIZE){
            int random_variable = random()% 10;
            if(random_variable< 4){
                breeding_population.push(copy_queue.top());
                copy_queue.pop();
                counter++;
            }
            if(copy_queue.size() == 0){
                copy_queue = selection_queue;
            }
        }
        /*
                printf("UNLOADING//////////\n");
        int x = 0;
        while(!breeding_population.empty()){
            path_t * temp = breeding_population.front();
            breeding_population.pop();
            printf("///////////%d\n",x++ );
            for(int i = 0; i < temp->path.size(); i++){
                printf("%d ",temp->path[i] );
            }
            printf("\n");
        }

        printf("///////////\n");
        return;

        */

        
        // 3.2. Second, we take two parents and make an offspring
        //cout << "now we have created our parent population, we're off to make babies\n";
        
        while(!path_queue.empty()){
            path_queue.pop();
        }
        //cout << "finished popping\n";
        std::vector<int > first_kid;
        std::vector<int > second_kid;
        while(!breeding_population.empty()){
            path_t * father = breeding_population.front();
            breeding_population.pop();
            if(breeding_population.empty())break;

            path_t * mother = breeding_population.front();
            breeding_population.pop();
            /*cout << "father:\n";
            for(int i = 0; i < father->path.size(); i++){
                printf("%d ", father->path[i]);
            }
            printf("\nmother:\n");
            for(int i = 0; i < mother->path.size(); i++){
                printf("%d ", mother->path[i]);
            }
            printf("\n");
            */

            first_kid = cross_over(father->path, mother->path, 1);
            /*printf("try to be here\n");
            for(int i = 0; i < first_kid.size(); i++){
                printf("%d ", first_kid[i]);
            }
            printf("\n");*/
            second_kid =cross_over(father->path, mother->path, 2);
            //cout << "we have finished breeding\n";

            // place children into the new population
            path_t * first_offspring = (path_t*)calloc(1, sizeof(path_t));
            path_t * second_offspring = (path_t*)calloc(1, sizeof(path_t));
            
            first_offspring->id = father->id;
            second_offspring->id = mother->id;



            first_kid = perturbate(first_kid);
            second_kid = perturbate(second_kid);

            //printf("so far so good...\n");
            //printf("evaluate first:%d, evaluate second:%d\n",evaluate(first_kid), evaluate(second_kid) );
            first_offspring->cost = evaluate(first_kid);
            second_offspring->cost = evaluate(second_kid);

            first_offspring->path = first_kid;
            second_offspring->path = second_kid;

            //population.push_back(first_child);
            //population.push_back(second_child);
            path_queue.push(first_offspring);
            path_queue.push(second_offspring);

        }
        printf("current generation:%d, best path cost:%d\n",i, path_queue.top()->cost  );
        result.push_back(path_queue.top()->path);
        for(int i = 0; i < path_queue.top()->path.size(); i++){
            printf("%d ",path_queue.top()->path[i] );
        }
        printf("\n");
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*while(!breeding_population.empty()){
        curr = breeding_population.front();
        breeding_population.pop();
        printf("%d\n",curr->cost );
    }*/
    return;
}


int main (){
    std::srand ( unsigned ( std::time(0) ) );
    load_data();
    if(ALGORITHM == 1){
        genetic_algorithm();
    }else if(ALGORITHM == 2){
        int solution = SA(random_solution());
    }else if(ALGORITHM == 3){
        memetic_genetic_algorithm();
    }else if(ALGORITHM == 4){ // nearest neighbour constructive heuristic
        nearest_neighbour(rand()%towns);
    }
    return 0;
}








