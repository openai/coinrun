from coinrun import random_agent

def test_coinrun():
    random_agent.random_agent(num_envs=16, max_steps=100)


if __name__ == '__main__':
    test_coinrun()