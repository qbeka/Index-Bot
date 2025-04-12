# Index Bot 

A Discord bot designed to form balanced teams for hackathons and competitions based on participants' skills, experience, and preferences.

## Features 

- **Smart Team Formation**: Uses a simmulated annealing algorithm to create balanced teams based on multiple factors
- **Participant Profiles**: Allows users to create detailed profiles with their skills and experience
- **Competition Management**: Create and manage different competitions with specific role requirements
- **Role-Based Team Building**: Ensures teams have the required roles and skills
- **Leader Assignment**: Automatically identifies potential team leaders
- **Private Team Channels**: Creates dedicated Discord channels for each team

## Installation 🛠

1. Clone the repository:
```bash
git clone https://github.com/qbeka/Index-Bot.git
cd Index-Bot
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Discord bot token:
```env
DISCORD_TOKEN=your_discord_token_here
```

## Usage 

1. Start the bot:
```bash
python bot.py
```

2. Bot Commands:
- `!create-profile` - Create your participant profile
- `!add-comp <name>` - Create a new competition (Admin only)
- `!join-comp <name>` - Join a competition
- `!list-participants <name>` - List participants in a competition
- `!make-teams <comp_name> <team_size>` - Form teams for a competition

## How It Works 

The bot uses an algorithm that considers:
- Hackathon scores
- Project experience
- GitHub contributions
- Experience level
- Problem-solving skills
- Innovation index
- Role requirements
- Leadership preferences

Teams are formed using a simulated annealing algorithm to ensure optimal balance and role distribution.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 

- Discord.py for the bot framework
- NumPy for mathematical operations
- Python-dotenv for environment variable management
