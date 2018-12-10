#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#define MIN_SAMPLE_SIZE   (7)
#define MIN_SEARCH_SIZE	  (7)
#define MAX_TREE_DEPTH    (27)
#define MAX_SEARCH_DEPTH  (27)
#define FOREST_SIZE	  (512)		//lets get 100 trees in one forest
#define SAMPLE_SIZE	  (42000)
#define N_FEATURES	  (12)		//number of features*28*28
#define FEATURE_SET	  (28)		//amount of features to look for, m = sqrt(# of input variables)
#define GRADIENT          (10)           //fit 10% of all images into each classification tree

class node;
node **forest;

class digit{
public:
	short *pixels;
	int class_id;
	digit(int class_id, short *pixels){
		this->pixels = pixels;
		this->class_id = class_id;
	}
	~digit(){
		free(pixels);
	}
};

class node{
public:
	int chosen_feature;
	short i, j, *feature;
	vector<digit*> arr;
	node *left, *right;
	node(){
		chosen_feature = -1;
		right = nullptr;
		left = nullptr;
		i = -1;
		j = -1;
		feature = (short*)malloc(sizeof(short)*N_FEATURES);
		for(int i = 1; i <= N_FEATURES; i++)
			feature[i-1] = (short)(((float)i/(N_FEATURES+1))*255.0);
	}
	~node(){
		arr.clear();
		free(feature);
		delete left;
		delete right;
	}
};

void init_tree(int tree_id, digit *dg){			//add an entry to tree
	if(forest[tree_id] == NULL){
		node *root = new node();
		root->arr.push_back(dg);
		forest[tree_id] = root;
	}
	else
		forest[tree_id]->arr.push_back(dg);
}

void grow_tree(node *root, short depth){
	//lets use gini to split node
	if(depth > MAX_TREE_DEPTH)
		return;
	float min_gini = 1.0f;
	#pragma acc kernels
	{
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){
				for(int ftr = 0; ftr < N_FEATURES; ftr++){
					float right_gini = 0, left_gini = 0, weighted_gini = 0;
					short *freq_left = (short*)malloc(sizeof(short)*10);
					short *freq_right = (short*)malloc(sizeof(short)*10);
					short n_right = 0, n_left = 0, n = root->arr.size();
					memset(freq_right,0,sizeof(short)*10);
					memset(freq_left,0,sizeof(short)*10);
					for(int k = 0; k < root->arr.size(); k++){
						digit *focus = root->arr.at(k);
						if(focus->pixels[i*28+j] > root->feature[ftr]){
							freq_right[focus->class_id]++;
							n_right++;
						}
						else{
							freq_left[focus->class_id]++;
							n_left++;
						}
					}
					for(int k = 0; k < 10; k++){
						right_gini += pow((freq_right[k]/(float)n_right),2);
						left_gini += pow((freq_left[k]/(float)n_left),2);
					}
					right_gini = 1 - right_gini;
					left_gini  = 1 - left_gini;
					weighted_gini = right_gini*((float)n_right/(float)n)
								  + left_gini*((float)n_left/(float)n);
					if(weighted_gini < min_gini && (n_right >= MIN_SAMPLE_SIZE || n_left >= MIN_SAMPLE_SIZE)){
						min_gini = weighted_gini;
						root->chosen_feature = ftr;
						root->i = i;
						root->j = j;
						//cout<<weighted_gini<<endl;
					}
					free(freq_left);
					free(freq_right);
				}
			}
		}
	}
	node *right = new node();
	node *left  = new node();
	for(int k = 0; k < root->arr.size(); k++){
		digit *focus = root->arr.at(k);
		if(focus->pixels[root->i*28+root->j] > root->feature[root->chosen_feature])
			right->arr.push_back(focus);
		else
			left->arr.push_back(focus);
	}
	//cout<<left->arr.size()<<" "<<right->arr.size()<<endl;
	if(right->arr.size() >= MIN_SAMPLE_SIZE && left->arr.size() >= MIN_SAMPLE_SIZE){
		grow_tree(right, depth+1);
		grow_tree(left, depth+1);
	}/*
	else{
		for(int i = 0; i < root->arr.size(); i++)
			cout<<root->arr.at(i)->class_id<<" ";
		cout<<endl;
	}*/
	root->right = right;
	root->left  = left;
}

void grow_random_tree(node *root, int depth){
	//lets use gini to split node
	if(depth > MAX_TREE_DEPTH)
		return;
	float min_gini = 1.0f;
	for(int x = 0; x < (int)sqrt(FEATURE_SET); x++){
		int i = rand()%FEATURE_SET;
		for(int y = 0; y < (int)sqrt(FEATURE_SET); y++){
			int j = rand()%FEATURE_SET;
			for(int ftrs = 0; ftrs < (int)sqrt(N_FEATURES); ftrs++){
				int ftr = rand()%N_FEATURES;
				float right_gini = 0, left_gini = 0, weighted_gini = 0;
				short *freq_left = (short*)malloc(sizeof(short)*10);
				short *freq_right = (short*)malloc(sizeof(short)*10);
				short n_right = 0, n_left = 0, n = root->arr.size();
				memset(freq_right,0,sizeof(short)*10);
				memset(freq_left,0,sizeof(short)*10);
				for(int k = 0; k < root->arr.size(); k++){
					digit *focus = root->arr.at(k);
					if(focus->pixels[i*28+j] > root->feature[ftr]){
						freq_right[focus->class_id]++;
						n_right++;
					}
					else{
						freq_left[focus->class_id]++;
						n_left++;
					}
				}
				for(int k = 0; k < 10; k++){
					right_gini += pow((freq_right[k]/(float)n_right), 2);
					left_gini += pow((freq_left[k]/(float)n_left), 2);
				}
				right_gini = 1 - right_gini;
				left_gini  = 1 - left_gini;
				weighted_gini = right_gini*((float)n_right/(float)n)
							  + left_gini*((float)n_left/(float)n);
				if(weighted_gini < min_gini && (n_right >= MIN_SAMPLE_SIZE || n_left >= MIN_SAMPLE_SIZE)){
					min_gini = weighted_gini;
					root->chosen_feature = ftr;
					root->i = i;
					root->j = j;
					//cout<<weighted_gini<<endl;
				}
				free(freq_left);
				free(freq_right);
			}
		}
	}
	node *right = new node();
	node *left  = new node();
	for(int k = 0; k < root->arr.size(); k++){
		digit *focus = root->arr.at(k);
		if(focus->pixels[root->i*28+root->j] > root->feature[root->chosen_feature])
			right->arr.push_back(focus);
		else
			left->arr.push_back(focus);
	}
	//cout<<left->arr.size()<<" "<<right->arr.size()<<endl;
	if(right->arr.size() >= MIN_SAMPLE_SIZE && left->arr.size() >= MIN_SAMPLE_SIZE){
		grow_tree(right, depth+1);
		grow_tree(left, depth+1);
	}/*
	else{
		for(int i = 0; i < root->arr.size(); i++)
			cout<<root->arr.at(i)->class_id<<" ";
		cout<<endl;
	}*/
	root->right = right;
	root->left  = left;
}

float* predict(node *root, short *dg, int depth){
	if(depth > MAX_SEARCH_DEPTH
		|| root->arr.size() < MIN_SEARCH_SIZE
		|| (root->right == nullptr && root->left == nullptr)
		|| root->i == -1
		|| root->j == -1
	){
        float *predictions = (float*)malloc(sizeof(float)*10);
        for(int i = 0; i < 10; i++)
            predictions[i] = 0.0;
        for(int k = 0; k < root->arr.size(); k++){
            digit *focus = root->arr.at(k);
            predictions[focus->class_id]+=1.0;
        }
        for(int i = 0; i < 10; i++)
            predictions[i] /= root->arr.size();
        return predictions;
    }
    if(dg[root->i*28+root->j] > root->feature[root->chosen_feature] && root->right != nullptr)
        return predict(root->right, dg, depth+1);
    else if(root->left != nullptr)
        return predict(root->left, dg, depth+1);
}

int bootstrap(float *prediction){
	float majority = 0;
	int final_prediction = -1;
	for(int i = 0; i < 10; i++){
		if(prediction[i] > majority){
			majority = prediction[i];
			final_prediction = i;
		}
	}
	return final_prediction;
}

int main(int argc, char **argv){
	//collect and visualize data
	forest = (node**)malloc(sizeof(node*)*FOREST_SIZE);		//50 null tree pointers
    time_t t;
    srand((unsigned) time(&t));
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
        short *gray_scale = (short*)malloc(sizeof(short)*(28*28));
        int class_id = tokens[0];
        for(int x = 0; x < 28; x ++){
            for(int y = 0; y < 28; y ++)
                gray_scale[x*28+y] = tokens[x*28+y+1];
        }
        digit *entry = new digit(class_id, gray_scale);
        #pragma omp parallel for
        for(int k = 0; k < FOREST_SIZE; k++) {
        	if (rand() % GRADIENT == 0)
        		init_tree(k, entry);
        }
        free(in);
    }
    int cntr = 0;
    #pragma omp parallel for
    for(int i = 0; i < FOREST_SIZE; i++){
        grow_random_tree(forest[i], 0);
        #pragma omp critical
        {
        	cout<<"tree "<<cntr<<" built."<<endl;
    		cntr++;
    	}
    }

    cout<<"-----------finished growing forest---------------"<<endl;

    //============================================== predictive testing ====================================================

    freopen("../test/test.csv","r",stdin);
    freopen("../test/sub1.csv","w",stdout);
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
        short *gray_scale = (short*)malloc(sizeof(short)*(28*28));            //we dont need to add one because classification data is not required for prediction
        for(int x = 0; x < 28; x++){
            for(int y = 0; y < 28; y++)
            	gray_scale[x*28+y] = tokens[x*28+y];
        }
        short final_prediction = -1;
        int *majority = (int*)malloc(sizeof(int)*10);
        int votes = 0;
        memset(majority,0,sizeof(int)*10);
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < FOREST_SIZE; i++){
        	int idx = bootstrap(predict(forest[i],gray_scale,0));
        	if(idx == -1)
        		continue;
        	#pragma omp critical
        	majority[idx]++;
        	#pragma omp critical
        	{
        		if(majority[idx] > votes){
        			votes = majority[idx];
        			final_prediction = idx;
        		}
        	}
        }
        printf("%d,%d\n",amt+1,final_prediction);
        free(in);
        free(majority);
        free(gray_scale);
    }
	return 0;
}
