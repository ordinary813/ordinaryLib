#include <linux/module.h>
#include <linux/kernel.h>

int init_module(void)
{
    printk("Hello world 1.\n");
    return 0;
}

void cleanup_module(void)
{
    printk("Goodbye world 1.\n");
}