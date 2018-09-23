#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
//avoid overfitting
#define MIN_SAMPLE_SIZE  (4)        //dont split if this value or under
#define MIN_SEARCH_SIZE  (3)        //dont search if under
#define MAX_TREE_DEPTH   (23)
#define MAX_SEARCH_DEPTH (21)
#define SAMPLE_SIZE      (28000)    //how many lines to read

class node;
//vector<node*> tree(10);                //stores roots for each tree
node **tree;

class node{
    public:
        int target, chosen_feature;
        short sample_size;            //sample size = sub_set.size()
        float p,q;                    //sample split n(catagory), p = success, q = failure
        short x,y;                    //which pixel to compare lightness
        short *feature;               //features, 0-255
        node *left, *right;
        vector<short*> sub_set;       //pixel field
        node(char target){
            chosen_feature = 0;
            sample_size = 0;
            q = 0;
            p = 0;
            x = 0;
            y = 0;
            right = nullptr;
            left = nullptr;
            this->target = target;
            feature = (short*)malloc(sizeof(short)*3);
            feature[0] = 53;
            feature[1] = 157;
            feature[2] = 223;
        }
        ~node(){
            for(short *st: sub_set)
                free(st);
            free(feature);
            delete left;
            delete right;
        }
};

void init_tree(int class_id, short* gray_scale){
    if(tree[class_id] == NULL){
        node *root = new node(class_id);
        root->sub_set.push_back(gray_scale);
        tree[class_id] = root;
    }
    else{
        tree[class_id]->sub_set.push_back(gray_scale);
    }
    tree[class_id]->sample_size++;
}

void grow_tree(node *root, short depth){
    if(depth > MAX_TREE_DEPTH)
        return;
    short x = 0, y = 0;
    double max_info_gain = 0, max_p_bright = 0, max_q_bright = 0, max_p_dark = 0, max_q_dark = 0;
    int chosen_feature = 0;
    //try and optimize max info gain for split
    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            for(int ftr = 0; ftr < 3; ftr++){             //specific feature, there are three in this case
                //probability of success and probability of failure
                double p_bright = 0, q_bright = 0, p_dark = 0, q_dark = 0, dark_entropy = 0, bright_entropy = 0, weighted_entropy = 0;
                double bright_total = 0, dark_total = 0;
                for(int k = 0; k < root->sub_set.size(); k++){
                    short *features = root->sub_set.at(k);
                    if(features[i*28+j+1] >= root->feature[ftr]){
                        if(features[0] == (short)root->target)
                            p_bright++;
                            //success probability for bright +1
                        bright_total++;
                    }
                    else if(features[i*28+j+1] < root->feature[ftr]){
                        if(features[0] == (short)root->target)
                            p_dark++;
                            //success probability for dark +1
                        dark_total++;
                    }
                }
                q_bright  = bright_total - p_bright;
                q_dark    = dark_total - p_dark;
                p_bright  = p_bright/bright_total;
                q_bright  = q_bright/bright_total;
                p_dark    = p_dark/dark_total;
                q_dark    = q_dark/dark_total;
                bright_entropy   = -p_bright*(log(p_bright)/log(2)) - q_bright*(log(q_bright)/log(2));
                dark_entropy     = -p_dark*(log(p_dark)/log(2)) - q_dark*(log(q_dark)/log(2));
                weighted_entropy = (bright_total/(double)root->sample_size)*bright_entropy + (dark_total/(double)root->sample_size)*dark_entropy;
                if(1.0-weighted_entropy > max_info_gain){
                    max_info_gain = 1.0-weighted_entropy;
                    x = i;
                    y = j;
                    root->x = i;
                    root->y = j;
                    max_p_bright = p_bright;                //update the children for further use in the prediction process
                    max_q_bright = q_bright;
                    max_p_dark   = p_dark;
                    max_q_dark   = q_dark;
                    chosen_feature = ftr;
                }
            }
        }
    }
    //split
    node *right = new node(root->target);                //above feature
    node *left    = new node(root->target);                //below feature
    for(int k = 0; k < root->sub_set.size(); k++){
        short *features = root->sub_set.at(k);
        if(features[x*28+y+1] > root->feature[chosen_feature]){
            right->sub_set.push_back(features);
            right->sample_size++;
        }
        else{
            left->sub_set.push_back(features);
            left->sample_size++;
        }
    }
    right->p = max_p_bright;
    right->q = max_q_bright;

    left->p = max_p_dark;
    left->q = max_q_dark;

    if(right->sample_size >= MIN_SAMPLE_SIZE && left->sample_size >= MIN_SAMPLE_SIZE){
        //cout<<left->sample_size<<" "<<right->sample_size<<endl;
        grow_tree(right, depth+1);
        grow_tree(left, depth+1);
        root->left = left;
        root->right = right;
    }
}

float predict(node *root, short *features, int depth){            //gives chance of image being a zero
    if(depth > MAX_SEARCH_DEPTH || root->sample_size < MIN_SEARCH_SIZE || (root->right == nullptr && root->left == nullptr))
        return (float)(root->p);
    short x = root->x;
    short y = root->y;
    if(features[28*x+y] >= root->feature[root->chosen_feature] && root->right != nullptr)
        return predict(root->right, features, depth+1);
    else if(root->left != nullptr)
        return predict(root->left, features, depth+1);
}

//write out the trained decision trees
void write_out(node *root, char *dir){
    //print format x, y, avg, p
}

int main(int argc, char **argv){
    tree = (node**)malloc(sizeof(node*)*10);
    //collect and visualize data
    freopen("../test/train.csv","r",stdin);
    std::string ignore_line;
    cin>>ignore_line;
    for(int amt = 0; amt < SAMPLE_SIZE; amt++){
        char *in = (char*)malloc(sizeof(char)*3136);
        try{
            scanf("%s",in);
            if(strlen(in) <= 10)
                return 1;
        } catch (...){
            return 1;
        }
        string tokenize_str(in);
        vector<int> tokens;
        stringstream ss(in);
        string string_builder = "";
        char curr;
        while(ss >> curr){
            string_builder += curr;
            if(ss.peek() == ','){
                //deliminator
                tokens.push_back(stoi(string_builder));
                string_builder = "";
                ss.ignore();
            }
        }
        if(string_builder.compare("") != 0)        //get last token
            tokens.push_back(stoi(string_builder));
        //turn 28x28 to 14x14
        short *gray_scale = (short*)malloc(sizeof(short)*(28*28+1));            //extra 1 is for the classification id
        int class_id = tokens[0];
        gray_scale[0] = class_id;                //first feature element is class id
        for(int x = 0; x < 28; x++){
            for(int y = 0; y < 28; y++){
                /*
                short avg4x4 = tokens[ x   *28+y+1  ];
                avg4x4      += tokens[(x+1)*28+y+1  ];
                avg4x4      += tokens[ x   *28+y+1+1];
                avg4x4      += tokens[(x+1)*28+y+1+1];
                gray_scale[(x/2)*14+(y/2)+1] = (short)(avg4x4/2);
                */
                gray_scale[x*28+y+1] = (short) tokens[x*28+y+1];
            }
        }
        #pragma omp parallel for
        for(int i = 0; i < 10; i++)
            init_tree(i, gray_scale);           //add entries to the different trees
        free(in);
    }
    //grow the tree, turns 28x28 into 14x14 gradient
    int th_count = (int)thread::hardware_concurrency();
    #pragma omp parallel for num_threads(th_count)
    for(int i = 0; i < 10 ; i++)
        grow_tree(tree[i],0);
    //============================================== predictive testing ====================================================
    freopen("../test/test.csv","r",stdin);
    freopen("../test/sub.csv","w",stdout);
    ignore_line;
    cin>>ignore_line;
    for(int amt = 0; amt < 28000; amt++){
        char *in = (char*)malloc(sizeof(char)*3136);
        try{
            scanf("%s",in);
            if(strlen(in) <= 10)
                return 1;
        } catch (...){
            return 1;
        }
        string tokenize_str(in);
        vector<int> tokens;
        stringstream ss(in);
        string string_builder = "";
        char curr;
        while(ss >> curr){
            string_builder += curr;
            if(ss.peek() == ','){
                //deliminator
                tokens.push_back(stoi(string_builder));
                string_builder = "";
                ss.ignore();
            }
        }
        if(string_builder.compare("") != 0)        //get last token
            tokens.push_back(stoi(string_builder));
        //turn 28x28 to 14x14
        short *gray_scale = (short*)malloc(sizeof(short)*(28*28));
        for(int x = 0; x < 28; x++){
            for(int y = 0; y < 28; y++){
                /*
                short avg4x4 = tokens[ x   *28+y  ];
                avg4x4      += tokens[(x+1)*28+y  ];
                avg4x4      += tokens[ x   *28+y+1];
                avg4x4      += tokens[(x+1)*28+y+1];
                gray_scale[(x/2)*14+(y/2)] = (short)(avg4x4/2);
                */
                gray_scale[x*28+y] = (short)tokens[x*28+y];
            }
        }/*
        printf("-- PICTURE %d --\n",amt);
        printf("prediction0: %f\n",predict( tree[0], gray_scale, 0));
        printf("prediction1: %f\n",predict( tree[1], gray_scale, 0));
        printf("prediction2: %f\n",predict( tree[2], gray_scale, 0));
        printf("prediction3: %f\n",predict( tree[3], gray_scale, 0));
        printf("prediction4: %f\n",predict( tree[4], gray_scale, 0));
        printf("prediction5: %f\n",predict( tree[5], gray_scale, 0));
        printf("prediction6: %f\n",predict( tree[6], gray_scale, 0));
        printf("prediction7: %f\n",predict( tree[7], gray_scale, 0));
        printf("prediction8: %f\n",predict( tree[8], gray_scale, 0));
        printf("prediction9: %f\n",predict( tree[9], gray_scale, 0));*/
        int prediction_num = -1;
        float max_pct = 0.00f;
        for(int i = 0; i < 10; i++){
            float tmp = predict(tree[i], gray_scale, 0);
            if(tmp > max_pct){
                max_pct = tmp;
                prediction_num = i;
            }
        }
        printf("%d,%d\n",(int)(amt+1),prediction_num);
        free(in);
    }
    free(tree);
    return 0;
}
