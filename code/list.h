#ifndef _LIST_H
#define _LIST_H

// 双向链表节点
struct list_head {
    struct list_head *next, *prev;
};

// 初始化节点：设置name节点的前继节点和后继节点都是指向name本身。
#define LIST_HEAD_INIT(name) { &(name), &(name) }

// 定义表头(节点)：新建双向链表表头name，并设置name的前继节点和后继节点都是指向name本身。
#define LIST_HEAD(name) \
    struct list_head name = LIST_HEAD_INIT(name)

// 初始化节点：将list节点的前继节点和后继节点都是指向list本身。
static inline void INIT_LIST_HEAD(struct list_head *list)
{
    list->next = list;
    list->prev = list;
}

// 添加节点：将new插入到prev和next之间。
static inline void __list_add(struct list_head *newnode,
                  struct list_head *prev,
                  struct list_head *next)
{
    next->prev = newnode;
    newnode->next = next;
    newnode->prev = prev;
    prev->next = newnode;
}

// 添加new节点：将new添加到head之后，是new称为head的后继节点。
static inline void list_add(struct list_head *newnode, struct list_head *head)
{
    __list_add(newnode, head, head->next);
}

// 添加new节点：将new添加到head之前，即将new添加到双链表的末尾。
static inline void list_add_tail(struct list_head *newnode, struct list_head *head)
{
    __list_add(newnode, head->prev, head);
}

// 从双链表中删除entry节点。
static inline void __list_del(struct list_head * prev, struct list_head * next)
{
#ifdef DEBUG
    fprintf(stderr,"in __list_del before prev \n");
#endif
    if(next == NULL && prev == NULL)
    {
        ;
    }
    else if(next != NULL && prev == NULL)
    {
        next->prev = prev;
    }
    else if(next == NULL && prev != NULL)
    {
        prev->next = next;
    }
    else{
        prev->next = next;
        next->prev = prev;
    }
#ifdef DEBUG
    fprintf(stderr,"finish __list_del \n");
#endif
}

// 从双链表中删除entry节点。
static inline void list_del(struct list_head *entry)
{
    __list_del(entry->prev, entry->next);
}


// 从双链表中删除entry节点。
static inline void __list_del_entry(struct list_head *entry)
{
    __list_del(entry->prev, entry->next);
}

// 从双链表中删除entry节点，并将entry节点的前继节点和后继节点都指向entry本身。
static inline void list_del_init(struct list_head *entry)
{
    __list_del_entry(entry);
    INIT_LIST_HEAD(entry);
#ifdef DEBUG
    fprintf(stderr," list_del_init finished \n");
#endif
}

static inline void list_del_init_tail(struct list_head *head)
{
	list_del_init(head->prev);
}

// 用new节点取代old节点
static inline void list_replace(struct list_head *old,
                struct list_head *newnode)
{
    newnode->next = old->next;
    newnode->next->prev = newnode;
    newnode->prev = old->prev;
    newnode->prev->next = newnode;
}

// 双链表是否为空
static inline int list_empty(const struct list_head *head)
{
    return head->next == head;
}

// 获取"MEMBER成员"在"结构体TYPE"中的位置偏移
#define offsetofelement(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

// 根据"结构体(type)变量"中的"域成员变量(member)的指针(ptr)"来获取指向整个结构体变量的指针
#define container_of(ptr, type, member) ({          \
    const decltype( ((type *)0)->member ) *__mptr = (ptr);    \
    (type *)( (char *)__mptr - offsetofelement(type,member) );})

// 遍历双向链表
#define list_for_each(pos, head) \
    for (pos = (head)->next; pos != (head); pos = pos->next)

#define list_for_each_safe(pos, n, head) \
    for (pos = (head)->next, n = pos->next; pos != (head); \
        pos = n, n = pos->next)

#define list_entry(ptr, type, member) \
    container_of(ptr, type, member)

#endif
