from fabric import task

from config.utils.role_based import build_group_list
from ai.utils.test import test_openai_manager

@task
def buildgrouplist(ctx):
    build_group_list()

@task
def testopenaimanager(ctx):
    test_openai_manager()