#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "concurrent_list.h"

struct node
{
  int value;
  struct node *next;
};

struct list
{
  struct node *head;
};

void print_node(node *node)
{
  if (node)
  {
    printf("%d ", node->value);
  }
}

list *create_list()
{
  struct list *l = malloc(sizeof *l);
  l->head = NULL;
  return l;
}

void delete_list(list *list)
{
  struct node *tmp = NULL;
  while (list->head != NULL)
  {
    tmp = list->head;
    list->head = list->head->next;
    free(tmp);
    tmp = NULL;
  }
}

void insert_value(list *list, int value)
{
  struct node *curr = list->head;
  while (curr->value < value && curr->next->value < value)
    curr = curr->next;

  struct node *ins = malloc(sizeof *ins);
  ins->value = value;

  if (curr->value < value)
  {
    ins->next = curr->next;
    curr->next = ins;
  }
  else if (curr->value > value)
  {
    ins->next = curr;
    curr = ins;
  }

  free(curr);
  curr = NULL;
}

void remove_value(list *list, int value)
{
  struct node *curr = list->head;
  if (curr->value = value)
  {
    list->head = curr->next;
    curr->next = NULL;

    free(curr);
    curr = NULL;
    return;
  }

  while (curr->next->value != value && curr != NULL)
    curr = curr->next;

  if (curr->next->value == value)
  {
    struct node *tmp = curr->next;
    curr->next = tmp->next;
    tmp->next = NULL;

    free(tmp);
    tmp = NULL;
    return;
  }
}

void print_list(list *list)
{
  struct node *curr = list->head;
  while (curr != NULL)
    print_node(curr);

  curr = NULL;
  free(curr);

  printf("\n"); // DO NOT DELETE
}

void count_list(list *list, int (*predicate)(int))
{
  int count = 0; // DO NOT DELETE

  struct node *curr = list->head;

  while (curr != NULL)
  {
    if (predicate(curr->value))
      count++;
  }

  printf("%d items were counted\n", count); // DO NOT DELETE
}

void main()
{
  // struct node *n2 = malloc(sizeof *n2);
  // struct node *n3 = malloc(sizeof *n3);

  // n2->value = 13;
  // n3->value = 15;

  struct node *n1 = malloc(sizeof *n1);
  n1->value = 10;

  printf("n1: ");
  print_node(n1);

  struct list *l = malloc(sizeof *l);
  l->head = n1;

  insert_value(l, 15);
  insert_value(l, 20);

  printf("list: ");
  print_list(l);
}
