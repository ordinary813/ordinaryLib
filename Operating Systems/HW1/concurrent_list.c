// #include <pthread.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <limits.h>
// #include "concurrent_list.h"

// struct node
// {
//   int value;
//   struct node *next;
// };

// struct list
// {
//   struct node *head;
// };

// void print_node(node *node)
// {
//   if (node)
//   {
//     printf("%d ", node->value);
//   }
// }

// list *create_list()
// {
//   struct list *l = malloc(sizeof *l);
//   l->head = NULL;
//   return l;
// }

// void delete_list(list *list)
// {
//   struct node *tmp = NULL;
//   while (list->head != NULL)
//   {
//     tmp = list->head;
//     list->head = list->head->next;
//     free(tmp);
//     tmp = NULL;
//   }
// }

// void insert_value(list *list, int value)
// {
//   struct node *curr = list->head;
//   while (curr->value < value && curr->next->value < value)
//     curr = curr->next;

//   struct node *ins = malloc(sizeof *ins);
//   ins->value = value;

//   if (curr->value < value)
//   {
//     ins->next = curr->next;
//     curr->next = ins;
//   }
//   else if (curr->value > value)
//   {
//     ins->next = curr;
//     curr = ins;
//   }

//   free(curr);
//   curr = NULL;
// }

// void remove_value(list *list, int value)
// {
//   struct node *curr = list->head;
//   if (curr->value = value)
//   {
//     list->head = curr->next;
//     curr->next = NULL;

//     free(curr);
//     curr = NULL;
//     return;
//   }

//   while (curr->next->value != value && curr != NULL)
//     curr = curr->next;

//   if (curr->next->value == value)
//   {
//     struct node *tmp = curr->next;
//     curr->next = tmp->next;
//     tmp->next = NULL;

//     free(tmp);
//     tmp = NULL;
//     return;
//   }
// }

// void print_list(list *list)
// {
//   struct node *curr = list->head;
//   while (curr != NULL)
//     print_node(curr);

//   curr = NULL;
//   free(curr);

//   printf("\n"); // DO NOT DELETE
// }

// void count_list(list *list, int (*predicate)(int))
// {
//   int count = 0; // DO NOT DELETE

//   struct node *curr = list->head;

//   while (curr != NULL)
//   {
//     if (predicate(curr->value))
//       count++;
//   }

//   printf("%d items were counted\n", count); // DO NOT DELETE
// }

// void main()
// {
//   // struct node *n2 = malloc(sizeof *n2);
//   // struct node *n3 = malloc(sizeof *n3);

//   // n2->value = 13;
//   // n3->value = 15;

//   struct node *n1 = malloc(sizeof *n1);
//   n1->value = 10;

//   printf("n1: ");
//   print_node(n1);

//   struct list *l = malloc(sizeof *l);
//   l->head = n1;

//   insert_value(l, 15);
//   insert_value(l, 20);

//   printf("list: ");
//   print_list(l);
// }

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "concurrent_list.h"

struct node {
  	int value;
	node* next;
	pthread_mutex_t lock;
};

struct list {
  	pthread_mutex_t lock;
	node* head;
};

void print_node(node* node)
{
  if(node)
  {
	pthread_mutex_lock(&(node->lock));
	printf("%d ", node->value);
	pthread_mutex_unlock(&(node->lock));
  }
}

list* create_list()
{
	//initialize the list
	list* list;
	list=(struct list*)malloc(sizeof(list));
	list->head=NULL;
	pthread_mutex_init(&(list->lock),NULL);
  	return list;
}

void delete_list(list* list)
{
	//if list doesn't exist
	if(!list)
		return;
	// if the head node doesn't exist
	if(!list->head)
	{
		pthread_mutex_destroy(&(list->lock));
		free(list);
		return;
	}
	node* temp1;
	node* temp2;
	temp1=list->head;

	// traverse entire list
	while(temp1->next)
	{
		temp2=temp1->next;
		pthread_mutex_destroy(&(temp1->lock));
		free(temp1);
		temp1=temp2;
	}

	// delete last node
	pthread_mutex_destroy(&(temp1->lock));
	free(temp1);

	// delete list 
	pthread_mutex_destroy(&(list->lock));
	free(list);
	return;
}

void insert_first(list* list, int val)
{
	list->head=(struct node*)malloc(sizeof(node));
	list->head->value=val;
	pthread_mutex_init(&(list->head->lock),NULL);
	list->head->next=NULL;
	return;
}

void insert_value(list* list, int value)
{
	node* temp;
	if(!list)
		return;

	pthread_mutex_lock(&(list->lock));
	if(!list->head)
	{
		insert_first(list,value);
		pthread_mutex_unlock(&(list->lock));
		return;
	}

	pthread_mutex_lock(&(list->head->lock));
	//head is lock so curr=head is locked
	node* curr=list->head;
	node* node_to_insert=(struct node*)malloc(sizeof(node));

	// initialize node for insertion
	node_to_insert->value=value;
	node_to_insert->next=NULL;
	pthread_mutex_init(&(node_to_insert->lock),NULL);

	// if the value is the smallest in the list - when it should be inserted before head
	if(curr->value>=value)
	{
		node_to_insert->next=curr;
		list->head=node_to_insert;
		pthread_mutex_unlock(&(curr->lock));
		pthread_mutex_unlock(&(list->lock));
		return;
	}
	pthread_mutex_unlock(&(list->lock));	

	// traverse list
	// in each iteration - lock the next node
	while(curr->next)
	{
		pthread_mutex_lock(&(curr->next->lock));
		// if the next value is still less than value
		// unlock the current node and go to the next node
		if(curr->next->value<value)
		{
			temp=curr->next;
			pthread_mutex_unlock(&(curr->lock));
			curr=temp;
		}
		// if the next node is not less than value
		// step out of loop
		else{
			break;
		}
	}

	// in case we need to add the new value to the end of the list
	// only curr is locked here
	if(!curr->next)
	{
		curr->next=node_to_insert;
		pthread_mutex_unlock(&(curr->lock));
		return;
	}

	// in case we insert it somewhere in the middle
	// in this case curr is locked and also the next node is locked
	node_to_insert->next=curr->next;
	curr->next=node_to_insert;
	pthread_mutex_unlock(&(node_to_insert->next->lock));
	pthread_mutex_unlock(&(curr->lock));
	return;
}

void remove_value(list* list, int value)
{
  	node* curr;
	node* temp;
	//pre-first check if list exists in first place
	if(!list)
		return;
	pthread_mutex_lock(&(list->lock));
	//first check that list isn't empty
	if(!list->head)
	{
		pthread_mutex_unlock(&(list->lock));
		return;
	}

	pthread_mutex_lock(&(list->head->lock));
	curr=list->head;
	// check if first node is to be removed
	if(curr->value==value)
	{
		list->head=list->head->next;
		pthread_mutex_unlock(&(curr->lock));
		pthread_mutex_destroy(&(curr->lock));
		free(curr);
		pthread_mutex_unlock(&(list->lock));
		return;
	}
	pthread_mutex_unlock(&(list->lock));

	//at this point curr=list->head is locked, list is unlocked
	// in each iteration we lock the next node
	while(curr->next)
	{
		pthread_mutex_lock(&(curr->next->lock));
		// if the next node contains the value - remove it
		if(curr->next->value==value)
		{
			temp=curr->next;
			curr->next=temp->next;
			pthread_mutex_unlock(&(curr->lock));
			pthread_mutex_unlock(&(temp->lock));
			pthread_mutex_destroy(&(temp->lock));
			free(temp);
			return;
		}

		// otherwise keep traversing the list
		temp = curr->next;
		pthread_mutex_unlock(&(curr->lock));
		curr = temp;
	}
	return;
}

void print_list(list* list)
{
	if(!list)
	{
		printf("\n ");
		return;
	}
	pthread_mutex_lock(&(list->lock));
	if(!list->head)
	{
		pthread_mutex_unlock(&(list->lock));
		printf("\n ");
		return;
	}

	node* curr;
	pthread_mutex_lock(&(list->head->lock));
	node* temp;
	curr=list->head;
	pthread_mutex_unlock(&(list->lock));

	while(curr->next)
	{
		printf("%d ",curr->value);
		temp=curr->next;
		pthread_mutex_unlock(&(curr->lock));
		curr=temp;
		pthread_mutex_lock(&(curr->lock));
	}
	printf("%d ",curr->value);
	pthread_mutex_unlock(&(curr->lock));
	printf("\n"); // DO NOT DELETE comment: okay
	return;
}

void count_list(list* list, int (*predicate)(int))
{
  	int count = 0; // DO NOT DELETE
	if(!list)
	{
		printf("\n ");
		return;
	}

	pthread_mutex_lock(&(list->lock));
	if(!list->head){
		printf("\n ");
		pthread_mutex_unlock(&(list->lock));
		return;
	}

	pthread_mutex_lock(&(list->head->lock));
	node* curr;
	node* temp;
	curr=list->head;
	pthread_mutex_unlock(&(list->lock));
	
	while(curr->next)
	{
		if(predicate(curr->value))
			count++;
		temp=curr->next;
		pthread_mutex_unlock(&(curr->lock));
		curr=temp;
		pthread_mutex_lock(&(curr->lock));
	}
	if(predicate(curr->value))
			count++;
	pthread_mutex_unlock(&(curr->lock));
  	printf("%d items were counted\n", count); // DO NOT DELETE okay please dont scream
	return;
}

